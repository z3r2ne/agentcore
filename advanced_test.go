package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestJSONSchemaAutomaticallyRejectsInvalidToolArguments(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "send", ArgumentsDelta: `{"count":"wrong"}`}}, StopReason: StopReasonToolUse}},
		{{TextDelta: "corrected", StopReason: StopReasonStop}},
	}}
	var executions atomic.Int32
	tool := FuncTool{
		ToolDefinition: ToolDefinition{
			Name: "send",
			Parameters: json.RawMessage(`{
				"type":"object",
				"properties":{"count":{"type":"integer"}},
				"required":["count"],
				"additionalProperties":false
			}`),
		},
		ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
			executions.Add(1)
			return TextToolResult("sent"), nil
		},
	}
	agent, err := New(Config{Model: model, Tools: []Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "send")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if executions.Load() != 0 {
		t.Fatal("schema-invalid tool was executed")
	}
	if message := result.State.Messages[2]; !message.IsError || message.Text() == "" {
		t.Fatalf("tool error message = %+v", message)
	}
}

func TestMissingToolCallIdentityAndArgumentsAreNormalized(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{{Index: 0, Name: "empty"}}, StopReason: StopReasonToolUse}},
		{{TextDelta: "done", StopReason: StopReasonStop}},
	}}
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "empty"}, ExecuteFunc: func(_ context.Context, arguments json.RawMessage, _ ToolUpdateSink) (ToolResult, error) {
		if string(arguments) != "{}" {
			t.Fatalf("arguments = %q", arguments)
		}
		return TextToolResult("ok"), nil
	}}
	agent, err := New(Config{Model: model, Tools: []Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	call := result.State.Messages[1].ToolCalls()[0]
	toolResult := result.State.Messages[2]
	if call.ID == "" || string(call.Arguments) != "{}" || toolResult.ToolCallID != call.ID {
		t.Fatalf("call=%+v tool result=%+v", call, toolResult)
	}
}

func TestToolInvocationMetadataIsStableAcrossRetries(t *testing.T) {
	model := toolThenStopModel("inspect")
	var invocations []ToolInvocation
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "inspect"}, ExecuteFunc: func(ctx context.Context, _ json.RawMessage, _ ToolUpdateSink) (ToolResult, error) {
		invocation, ok := ToolInvocationFromContext(ctx)
		if !ok {
			t.Fatal("tool invocation missing from context")
		}
		invocations = append(invocations, invocation)
		if invocation.Attempt == 1 {
			return ToolResult{}, errors.New("retry")
		}
		return TextToolResult("ok"), nil
	}}
	agent, err := New(Config{Model: model, Tools: []Tool{tool}, ToolPolicy: ToolPolicy{MaxAttempts: 2, RetryDelay: time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil); err != nil {
		t.Fatal(err)
	}
	if len(invocations) != 2 || invocations[0].Call.ID != "call-1" || invocations[1].Call.ID != invocations[0].Call.ID || invocations[0].Turn != 1 || invocations[1].Attempt != 2 || invocations[1].MaxAttempts != 2 {
		t.Fatalf("invocations = %+v", invocations)
	}
}

type retryModel struct {
	calls atomic.Int32
}

func (m *retryModel) Stream(context.Context, ModelRequest) (ModelStream, error) {
	if m.calls.Add(1) == 1 {
		return nil, errors.New("temporary transport failure")
	}
	return &sliceModelStream{chunks: []ModelChunk{{TextDelta: "ok", StopReason: StopReasonStop}}}, nil
}

type failAfterChunkStream struct {
	delivered bool
}

func (s *failAfterChunkStream) Recv() (ModelChunk, error) {
	if !s.delivered {
		s.delivered = true
		return ModelChunk{TextDelta: "partial"}, nil
	}
	return ModelChunk{}, errors.New("stream disconnected")
}

func (*failAfterChunkStream) Close() error { return nil }

type receiveRetryModel struct {
	calls atomic.Int32
}

func (m *receiveRetryModel) Stream(context.Context, ModelRequest) (ModelStream, error) {
	if m.calls.Add(1) == 1 {
		return &failAfterChunkStream{}, nil
	}
	return &sliceModelStream{chunks: []ModelChunk{{TextDelta: "complete", StopReason: StopReasonStop}}}, nil
}

func TestModelRetryEmitsPiStyleLifecycleEvents(t *testing.T) {
	model := &retryModel{}
	agent, err := New(Config{Model: model, ModelRetry: RetryPolicy{MaxAttempts: 2, InitialDelay: time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	var retryEvents []Event
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, func(_ context.Context, event Event) error {
		if event.Type == EventAutoRetryStart || event.Type == EventAutoRetryEnd {
			retryEvents = append(retryEvents, event)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if model.calls.Load() != 2 || result.State.Messages[1].Text() != "ok" {
		t.Fatalf("calls=%d result=%+v", model.calls.Load(), result)
	}
	if len(retryEvents) != 2 || retryEvents[0].Type != EventAutoRetryStart || retryEvents[0].Attempt != 2 || retryEvents[1].Type != EventAutoRetryEnd || !retryEvents[1].Success {
		t.Fatalf("retry events = %+v", retryEvents)
	}
}

func TestModelRetryAfterPartialStreamFailure(t *testing.T) {
	model := &receiveRetryModel{}
	agent, err := New(Config{Model: model, ModelRetry: RetryPolicy{MaxAttempts: 2, InitialDelay: time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if model.calls.Load() != 2 || result.State.Messages[len(result.State.Messages)-1].Text() != "complete" {
		t.Fatalf("calls=%d result=%+v", model.calls.Load(), result)
	}
	for _, message := range result.State.Messages {
		if message.Text() == "partial" {
			t.Fatal("failed partial response was persisted")
		}
	}
}

func TestModelPanicIsRetriedAsTransportFailure(t *testing.T) {
	model := &panicThenSucceedModel{}
	agent, err := New(Config{Model: model, ModelRetry: RetryPolicy{MaxAttempts: 2, InitialDelay: time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil)
	if err != nil || result.State.Messages[len(result.State.Messages)-1].Text() != "safe" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

type panicThenSucceedModel struct {
	calls atomic.Int32
}

func (m *panicThenSucceedModel) Stream(context.Context, ModelRequest) (ModelStream, error) {
	if m.calls.Add(1) == 1 {
		panic("provider bug")
	}
	return &sliceModelStream{chunks: []ModelChunk{{TextDelta: "safe", StopReason: StopReasonStop}}}, nil
}

func TestToolPolicyRetriesRecoversPanicAndTimesOut(t *testing.T) {
	t.Run("retry", func(t *testing.T) {
		model := toolThenStopModel("retry")
		var attempts atomic.Int32
		tool := FuncTool{ToolDefinition: ToolDefinition{Name: "retry"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
			if attempts.Add(1) == 1 {
				return ToolResult{}, errors.New("temporary")
			}
			return TextToolResult("ok"), nil
		}}
		agent, _ := New(Config{Model: model, Tools: []Tool{tool}, ToolPolicy: ToolPolicy{MaxAttempts: 2, RetryDelay: time.Millisecond}})
		var end Event
		_, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, func(_ context.Context, event Event) error {
			if event.Type == EventToolExecutionEnd {
				end = event
			}
			return nil
		})
		if err != nil || attempts.Load() != 2 || end.Attempt != 2 || end.IsError {
			t.Fatalf("attempts=%d end=%+v err=%v", attempts.Load(), end, err)
		}
	})

	t.Run("panic", func(t *testing.T) {
		model := toolThenStopModel("panic")
		tool := FuncTool{ToolDefinition: ToolDefinition{Name: "panic"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
			panic("boom")
		}}
		agent, _ := New(Config{Model: model, Tools: []Tool{tool}})
		result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if message := result.State.Messages[2]; !message.IsError || message.Text() == "" {
			t.Fatalf("panic result = %+v", message)
		}
	})

	t.Run("timeout", func(t *testing.T) {
		model := toolThenStopModel("wait")
		tool := FuncTool{ToolDefinition: ToolDefinition{Name: "wait"}, ExecuteFunc: func(ctx context.Context, _ json.RawMessage, _ ToolUpdateSink) (ToolResult, error) {
			<-ctx.Done()
			return ToolResult{}, ctx.Err()
		}}
		agent, _ := New(Config{Model: model, Tools: []Tool{tool}, ToolPolicy: ToolPolicy{Timeout: 5 * time.Millisecond}})
		result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if message := result.State.Messages[2]; !message.IsError {
			t.Fatalf("timeout result = %+v", message)
		}
	})
}

func TestMaxToolConcurrencyIsEnforced(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{
			{Index: 0, ID: "1", Name: "one", ArgumentsDelta: `{}`},
			{Index: 1, ID: "2", Name: "two", ArgumentsDelta: `{}`},
			{Index: 2, ID: "3", Name: "three", ArgumentsDelta: `{}`},
		}, StopReason: StopReasonToolUse}},
		{{TextDelta: "done", StopReason: StopReasonStop}},
	}}
	var active atomic.Int32
	var maximum atomic.Int32
	makeTool := func(name string) Tool {
		return FuncTool{ToolDefinition: ToolDefinition{Name: name}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
			current := active.Add(1)
			for {
				seen := maximum.Load()
				if current <= seen || maximum.CompareAndSwap(seen, current) {
					break
				}
			}
			time.Sleep(10 * time.Millisecond)
			active.Add(-1)
			return TextToolResult(name), nil
		}}
	}
	agent, _ := New(Config{Model: model, Tools: []Tool{makeTool("one"), makeTool("two"), makeTool("three")}, MaxToolConcurrency: 2})
	if _, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil); err != nil {
		t.Fatal(err)
	}
	if maximum.Load() != 2 {
		t.Fatalf("maximum concurrency = %d", maximum.Load())
	}
}

func TestPrepareNextTurnCanSwitchModelAndSystemPrompt(t *testing.T) {
	first := &fakeModel{responses: [][]ModelChunk{{{
		ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "next", ArgumentsDelta: `{}`}},
		StopReason:     StopReasonToolUse,
	}}}}
	second := &fakeModel{responses: [][]ModelChunk{{{TextDelta: "second", StopReason: StopReasonStop}}}}
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "next"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
		return TextToolResult("ok"), nil
	}}
	system := "new system"
	options := map[string]any{"thinking": "high", "credential": "dynamic"}
	agent, _ := New(Config{
		Model: first, Tools: []Tool{tool},
		Hooks: Hooks{PrepareNextTurn: func(_ context.Context, turn *TurnContext) error {
			turn.Next = &NextTurnConfig{Model: second, SystemPrompt: &system, ModelOptions: options}
			return nil
		}},
	})
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.State.Messages[len(result.State.Messages)-1].Text() != "second" || len(second.requests) != 1 || second.requests[0].SystemPrompt != system || !reflect.DeepEqual(second.requests[0].Options, options) {
		t.Fatalf("result=%+v requests=%+v", result, second.requests)
	}
}

func TestAfterModelHookInvalidatesStaleProviderData(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "next", ArgumentsDelta: `{}`}}, StopReason: StopReasonToolUse, ProviderData: &ProviderData{Format: "fake/v1", Data: json.RawMessage(`{"old":true}`)}}},
		{{TextDelta: "done", StopReason: StopReasonStop}},
	}}
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "next"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
		return TextToolResult("ok"), nil
	}}
	agent, err := New(Config{
		Model: model, Tools: []Tool{tool},
		Hooks: Hooks{AfterModelCall: func(_ context.Context, message *Message) error {
			if len(message.ToolCalls()) > 0 {
				message.Content = append(message.Content, ContentBlock{Type: ContentText, Text: "hook annotation"})
			}
			return nil
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, nil); err != nil {
		t.Fatal(err)
	}
	if got := model.requests[1].Messages[1].ProviderData; got != nil {
		t.Fatalf("stale provider data survived hook rewrite: %+v", got)
	}
}

func TestContextPolicyCompactsAndTruncatesBeforeModelCall(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{{{TextDelta: "done", StopReason: StopReasonStop}}}}
	compacted := false
	agent, err := New(Config{
		Model: model,
		ContextPolicy: ContextPolicy{
			MaxTokens:      25,
			EstimateTokens: func(messages []Message) int { return len(messages) * 10 },
			Compact: func(_ context.Context, messages []Message, target int) ([]Message, error) {
				compacted = true
				if target != 25 {
					t.Fatalf("target = %d", target)
				}
				if len(messages[1].Text()) > 24 || messages[1].Text() == "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" {
					t.Fatalf("tool result was not truncated: %q", messages[1].Text())
				}
				return messages[len(messages)-2:], nil
			},
			MaxToolResultBytes: 24,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	state := State{Messages: []Message{
		TextMessage(RoleUser, "old"),
		{Role: RoleTool, ToolCallID: "old-call", Content: []ContentBlock{{Type: ContentText, Text: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}}},
		TextMessage(RoleUser, "recent"),
	}}
	var compactEvents []EventType
	_, err = agent.Prompt(context.Background(), state, []Message{TextMessage(RoleUser, "new")}, func(_ context.Context, event Event) error {
		if event.Type == EventContextCompactStart || event.Type == EventContextCompactEnd {
			compactEvents = append(compactEvents, event.Type)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !compacted || !reflect.DeepEqual(compactEvents, []EventType{EventContextCompactStart, EventContextCompactEnd}) {
		t.Fatalf("compacted=%v events=%v", compacted, compactEvents)
	}
	if len(model.requests) != 1 || len(model.requests[0].Messages) != 2 {
		t.Fatalf("request = %+v", model.requests)
	}
}

func TestToolResultTruncationOmitsOversizedBinaryPayload(t *testing.T) {
	messages := []Message{{
		Role:    RoleTool,
		Content: []ContentBlock{{Type: ContentImage, Data: []byte("oversized-image"), MIMEType: "image/png"}},
	}}
	if !truncateToolResults(messages, 4) {
		t.Fatal("expected truncation")
	}
	block := messages[0].Content[0]
	if block.Type != ContentText || len(block.Data) != 0 || block.Text == "" || len(block.Text) > 4 {
		t.Fatalf("truncated block = %+v", block)
	}
}

func TestSummaryCompactorUsesModelAndKeepsRecentMessages(t *testing.T) {
	summaryModel := &fakeModel{responses: [][]ModelChunk{{{TextDelta: "important decision", StopReason: StopReasonStop}}}}
	compact, err := NewSummaryCompactor(SummaryCompactorConfig{
		Model: summaryModel, KeepRecent: 2,
		EstimateTokens: func(messages []Message) int { return len(messages) * 10 },
	})
	if err != nil {
		t.Fatal(err)
	}
	messages := []Message{
		TextMessage(RoleUser, "old question"),
		TextMessage(RoleAssistant, "old answer"),
		TextMessage(RoleUser, "recent question"),
		TextMessage(RoleAssistant, "recent answer"),
	}
	compacted, err := compact(context.Background(), messages, 30)
	if err != nil {
		t.Fatal(err)
	}
	if len(compacted) != 3 || compacted[0].Role != RoleUser || compacted[0].Text() != "[Earlier conversation summary]\nimportant decision" || compacted[2].Text() != "recent answer" {
		t.Fatalf("compacted = %+v", compacted)
	}
}

type controlledSessionModel struct {
	mu       sync.Mutex
	calls    int
	requests []ModelRequest
	started  chan struct{}
	release  chan struct{}
}

type blockingSessionModel struct {
	started chan struct{}
}

type failingSessionStore struct{}

func (failingSessionStore) SaveSession(context.Context, string, SessionSnapshot) error {
	return errors.New("disk full")
}

func (m *blockingSessionModel) Stream(ctx context.Context, _ ModelRequest) (ModelStream, error) {
	close(m.started)
	return &blockingModelStream{ctx: ctx}, nil
}

type blockingModelStream struct {
	ctx context.Context
}

func (s *blockingModelStream) Recv() (ModelChunk, error) {
	<-s.ctx.Done()
	return ModelChunk{}, s.ctx.Err()
}

func (*blockingModelStream) Close() error { return nil }

func (m *controlledSessionModel) Stream(_ context.Context, request ModelRequest) (ModelStream, error) {
	m.mu.Lock()
	m.calls++
	call := m.calls
	m.requests = append(m.requests, request)
	m.mu.Unlock()
	if call == 1 {
		close(m.started)
		<-m.release
	}
	return &sliceModelStream{chunks: []ModelChunk{{TextDelta: "turn", StopReason: StopReasonStop}}}, nil
}

func TestPiConformanceSteeringThenFollowUpDelivery(t *testing.T) {
	model := &controlledSessionModel{started: make(chan struct{}), release: make(chan struct{})}
	agent, _ := New(Config{Model: model})
	session, err := NewSession(agent, State{}, SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	stream := session.Stream(context.Background(), []Message{TextMessage(RoleUser, "initial")})
	var subscribed atomic.Int32
	unsubscribe := session.Subscribe(func(context.Context, Event) error {
		subscribed.Add(1)
		return nil
	})
	defer unsubscribe()
	<-model.started
	if !session.Status().Running {
		t.Fatal("session should be running")
	}
	if err := session.Steer(TextMessage(RoleUser, "steer-1"), TextMessage(RoleUser, "steer-2")); err != nil {
		t.Fatal(err)
	}
	if err := session.FollowUp(TextMessage(RoleUser, "follow")); err != nil {
		t.Fatal(err)
	}
	close(model.release)
	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 4 {
		t.Fatalf("turns = %d", result.Turns)
	}
	var users []string
	for _, message := range result.State.Messages {
		if message.Role == RoleUser {
			users = append(users, message.Text())
		}
	}
	if !reflect.DeepEqual(users, []string{"initial", "steer-1", "steer-2", "follow"}) {
		t.Fatalf("users = %v", users)
	}
	if err := session.WaitForIdle(context.Background()); err != nil || session.Status().Running {
		t.Fatalf("idle err=%v status=%+v", err, session.Status())
	}
	if subscribed.Load() == 0 {
		t.Fatal("persistent subscriber received no events")
	}
}

func TestSessionDeliveryAllInjectsEachQueueAsOneTurn(t *testing.T) {
	model := &controlledSessionModel{started: make(chan struct{}), release: make(chan struct{})}
	agent, _ := New(Config{Model: model})
	session, err := NewSession(agent, State{}, SessionOptions{SteeringMode: DeliveryAll, FollowUpMode: DeliveryAll})
	if err != nil {
		t.Fatal(err)
	}
	stream := session.Stream(context.Background(), []Message{TextMessage(RoleUser, "initial")})
	<-model.started
	if err := session.Steer(TextMessage(RoleUser, "steer-1"), TextMessage(RoleUser, "steer-2")); err != nil {
		t.Fatal(err)
	}
	if err := session.FollowUp(TextMessage(RoleUser, "follow-1"), TextMessage(RoleUser, "follow-2")); err != nil {
		t.Fatal(err)
	}
	close(model.release)
	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 3 {
		t.Fatalf("turns = %d", result.Turns)
	}
	var users []string
	for _, message := range result.State.Messages {
		if message.Role == RoleUser {
			users = append(users, message.Text())
		}
	}
	if !reflect.DeepEqual(users, []string{"initial", "steer-1", "steer-2", "follow-1", "follow-2"}) {
		t.Fatalf("users = %v", users)
	}
}

func TestSessionAbortCancelsActiveModelAndSettles(t *testing.T) {
	model := &blockingSessionModel{started: make(chan struct{})}
	agent, _ := New(Config{Model: model})
	session, _ := NewSession(agent, State{}, SessionOptions{})
	stream := session.Stream(context.Background(), []Message{TextMessage(RoleUser, "block")})
	<-model.started
	if err := session.Abort(); err != nil {
		t.Fatal(err)
	}
	result, err := stream.Result()
	if !errors.Is(err, context.Canceled) || result.StopReason != StopReasonAborted {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if err := session.WaitForIdle(context.Background()); err != nil || session.Status().Running {
		t.Fatalf("status=%+v err=%v", session.Status(), err)
	}
}

func TestSessionRestoreRepairsInterruptedParallelToolCallHistory(t *testing.T) {
	agent, err := New(Config{Model: &fakeModel{}})
	if err != nil {
		t.Fatal(err)
	}
	snapshot := SessionSnapshot{State: State{Messages: []Message{
		TextMessage(RoleUser, "start"),
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: ContentToolCall, ToolCall: &ToolCall{ID: "call-1", Name: "first", Arguments: json.RawMessage(`{}`)}},
			{Type: ContentToolCall, ToolCall: &ToolCall{ID: "call-2", Name: "second", Arguments: json.RawMessage(`{}`)}},
		}},
		{Role: RoleTool, ToolCallID: "call-1", ToolName: "first", Content: []ContentBlock{{Type: ContentText, Text: "done"}}},
		TextMessage(RoleUser, "resume after restart"),
	}}}

	session, err := NewSessionFromSnapshot(agent, snapshot, SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	messages := session.State().Messages
	if len(messages) != 5 {
		t.Fatalf("messages = %+v", messages)
	}
	if messages[2].ToolCallID != "call-1" || messages[2].IsError {
		t.Fatalf("existing result changed: %+v", messages[2])
	}
	if messages[3].Role != RoleTool || messages[3].ToolCallID != "call-2" || messages[3].ToolName != "second" || !messages[3].IsError {
		t.Fatalf("repaired result = %+v", messages[3])
	}
	if messages[4].Role != RoleUser || messages[4].Text() != "resume after restart" {
		t.Fatalf("later message moved incorrectly: %+v", messages[4])
	}
	if got := len(snapshot.State.Messages); got != 4 {
		t.Fatalf("input snapshot mutated: %d messages", got)
	}
}

func TestSessionRestoreKeepsCompleteToolCallHistoryUnchanged(t *testing.T) {
	agent, err := New(Config{Model: &fakeModel{}})
	if err != nil {
		t.Fatal(err)
	}
	snapshot := SessionSnapshot{State: State{Messages: []Message{
		{Role: RoleAssistant, Content: []ContentBlock{{Type: ContentToolCall, ToolCall: &ToolCall{ID: "call-1", Name: "done", Arguments: json.RawMessage(`{}`)}}}},
		{Role: RoleTool, ToolCallID: "call-1", ToolName: "done", Content: []ContentBlock{{Type: ContentText, Text: "ok"}}},
	}}}
	session, err := NewSessionFromSnapshot(agent, snapshot, SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if messages := session.State().Messages; len(messages) != 2 || messages[1].Text() != "ok" || messages[1].IsError {
		t.Fatalf("complete history changed: %+v", messages)
	}
}

func TestSessionCheckpointFailureIsReturnedAndStateRemainsAvailable(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{{{TextDelta: "done", StopReason: StopReasonStop}}}}
	agent, _ := New(Config{Model: model})
	session, err := NewSession(agent, State{}, SessionOptions{Store: failingSessionStore{}, SessionID: "failed"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := session.Prompt(context.Background(), []Message{TextMessage(RoleUser, "go")}, nil)
	if err == nil || !strings.Contains(err.Error(), "save session checkpoint") {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if len(session.State().Messages) != 2 || session.Status().LastError == "" || session.Status().Running {
		t.Fatalf("state=%+v status=%+v", session.State(), session.Status())
	}
}

func toolThenStopModel(toolName string) *fakeModel {
	return &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: toolName, ArgumentsDelta: `{}`}}, StopReason: StopReasonToolUse}},
		{{TextDelta: "done", StopReason: StopReasonStop}},
	}}
}
