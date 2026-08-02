package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeModel struct {
	mu        sync.Mutex
	responses [][]ModelChunk
	requests  []ModelRequest
}

func (m *fakeModel) Stream(_ context.Context, request ModelRequest) (ModelStream, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, request)
	if len(m.responses) == 0 {
		return nil, errors.New("unexpected model call")
	}
	chunks := m.responses[0]
	m.responses = m.responses[1:]
	return &sliceModelStream{chunks: chunks}, nil
}

type sliceModelStream struct {
	chunks []ModelChunk
	index  int
}

func (s *sliceModelStream) Recv() (ModelChunk, error) {
	if s.index >= len(s.chunks) {
		return ModelChunk{}, io.EOF
	}
	chunk := s.chunks[s.index]
	s.index++
	return chunk, nil
}

func (s *sliceModelStream) Close() error { return nil }

func TestPiConformanceToolLoopAndEventOrder(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{
			{TextDelta: "checking "},
			{ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "add", ArgumentsDelta: `{"a":`}}},
			{ToolCallDeltas: []ToolCallDelta{{Index: 0, ArgumentsDelta: `1,"b":2}`}}, StopReason: StopReasonToolUse},
		},
		{{TextDelta: "three"}, {StopReason: StopReasonStop, Usage: &Usage{OutputTokens: 1}}},
	}}
	tool := FuncTool{
		ToolDefinition: ToolDefinition{Name: "add", Parameters: json.RawMessage(`{"type":"object"}`)},
		ExecuteFunc: func(_ context.Context, arguments json.RawMessage, update ToolUpdateSink) (ToolResult, error) {
			if string(arguments) != `{"a":1,"b":2}` {
				t.Fatalf("arguments = %s", arguments)
			}
			if err := update(TextToolResult("working")); err != nil {
				return ToolResult{}, err
			}
			return TextToolResult("3"), nil
		},
	}
	agent, err := New(Config{Model: model, SystemPrompt: "be useful", Tools: []Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	var events []Event
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "add")}, func(_ context.Context, event Event) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 2 || result.StopReason != StopReasonStop {
		t.Fatalf("result = %+v", result)
	}
	if result.Usage.OutputTokens != 1 {
		t.Fatalf("usage = %+v", result.Usage)
	}
	if cost := result.Usage.Cost(Pricing{Currency: "USD", Output: 10}); cost.Currency != "USD" || cost.Amount != 0.00001 {
		t.Fatalf("cost = %+v", cost)
	}
	if len(result.State.Messages) != 4 {
		t.Fatalf("messages = %d", len(result.State.Messages))
	}
	if got := result.State.Messages[3].Text(); got != "three" {
		t.Fatalf("final text = %q", got)
	}
	if got := result.State.Messages[2]; got.Role != RoleTool || got.ToolCallID != "call-1" || got.Text() != "3" {
		t.Fatalf("tool message = %+v", got)
	}
	if len(model.requests) != 2 || model.requests[0].SystemPrompt != "be useful" {
		t.Fatalf("requests = %+v", model.requests)
	}
	if got := model.requests[1].Messages[len(model.requests[1].Messages)-1].Role; got != RoleTool {
		t.Fatalf("second request ended in role %q", got)
	}

	types := eventTypes(events)
	want := []EventType{
		EventAgentStart, EventTurnStart,
		EventMessageStart, EventMessageEnd,
		EventMessageStart, EventMessageUpdate, EventMessageUpdate, EventMessageUpdate, EventMessageEnd,
		EventToolExecutionStart, EventToolExecutionUpdate, EventToolExecutionEnd,
		EventMessageStart, EventMessageEnd, EventTurnEnd,
		EventTurnStart, EventMessageStart, EventMessageUpdate, EventMessageUpdate, EventMessageEnd,
		EventTurnEnd, EventAgentEnd,
	}
	if !reflect.DeepEqual(types, want) {
		t.Fatalf("event types:\n got  %v\n want %v", types, want)
	}
}

func TestStreamingToolArgumentsRemainSerializableDeltasUntilMessageEnd(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{
			{ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "open", ArgumentsDelta: `{`}}},
			{ToolCallDeltas: []ToolCallDelta{{Index: 0, ArgumentsDelta: `"path":`}}},
			{ToolCallDeltas: []ToolCallDelta{{Index: 0, ArgumentsDelta: `"README.md"}`}}, StopReason: StopReasonToolUse},
		},
		{{TextDelta: "done", StopReason: StopReasonStop}},
	}}
	tool := FuncTool{
		ToolDefinition: ToolDefinition{Name: "open", Parameters: json.RawMessage(`{"type":"object"}`)},
		ExecuteFunc: func(_ context.Context, arguments json.RawMessage, _ ToolUpdateSink) (ToolResult, error) {
			if string(arguments) != `{"path":"README.md"}` {
				t.Fatalf("arguments = %s", arguments)
			}
			return TextToolResult("opened"), nil
		},
	}
	agent, err := New(Config{Model: model, Tools: []Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	var deltas []string
	var completed ToolCall
	_, err = agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "open")}, func(_ context.Context, event Event) error {
		if _, err := json.Marshal(event); err != nil {
			return err
		}
		if event.Type == EventMessageUpdate && event.Delta != nil {
			for _, delta := range event.Delta.ToolCallDeltas {
				deltas = append(deltas, delta.ArgumentsDelta)
			}
			if event.Message != nil {
				for _, call := range event.Message.ToolCalls() {
					if len(call.Arguments) != 0 {
						t.Fatalf("partial message exposed final arguments: %s", call.Arguments)
					}
				}
			}
		}
		if event.Type == EventMessageEnd && event.Message != nil {
			calls := event.Message.ToolCalls()
			if len(calls) == 1 && calls[0].Name == "open" {
				completed = calls[0]
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(deltas, ""); got != `{"path":"README.md"}` {
		t.Fatalf("argument deltas = %q", got)
	}
	if string(completed.Arguments) != `{"path":"README.md"}` {
		t.Fatalf("completed call = %+v", completed)
	}
}

func TestMalformedCompletedToolArgumentsFailWithoutBreakingEventJSON(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{{{
		ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "broken", ArgumentsDelta: `{"path":`}},
		StopReason:     StopReasonToolUse,
	}}}}
	var executed atomic.Bool
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "broken"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
		executed.Store(true)
		return TextToolResult("unexpected"), nil
	}}
	agent, err := New(Config{Model: model, Tools: []Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "break")}, func(_ context.Context, event Event) error {
		_, marshalErr := json.Marshal(event)
		return marshalErr
	})
	if err == nil || !strings.Contains(err.Error(), "invalid JSON arguments") {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if executed.Load() {
		t.Fatal("malformed tool call was executed")
	}
}

func TestPiConformanceParallelCompletionAndPersistenceOrder(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{
			{Index: 0, ID: "slow-call", Name: "slow", ArgumentsDelta: `{}`},
			{Index: 1, ID: "fast-call", Name: "fast", ArgumentsDelta: `{}`},
		}, StopReason: StopReasonToolUse}},
		{{TextDelta: "done", StopReason: StopReasonStop}},
	}}
	slowStarted := make(chan struct{})
	fastDone := make(chan struct{})
	slow := FuncTool{ToolDefinition: ToolDefinition{Name: "slow"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
		close(slowStarted)
		<-fastDone
		return TextToolResult("slow-result"), nil
	}}
	fast := FuncTool{ToolDefinition: ToolDefinition{Name: "fast"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
		<-slowStarted
		close(fastDone)
		return TextToolResult("fast-result"), nil
	}}
	agent, err := New(Config{Model: model, Tools: []Tool{slow, fast}})
	if err != nil {
		t.Fatal(err)
	}
	var ended []string
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "run")}, func(_ context.Context, event Event) error {
		if event.Type == EventToolExecutionEnd {
			ended = append(ended, event.ToolName)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ended, []string{"fast", "slow"}) {
		t.Fatalf("completion order = %v", ended)
	}
	if got := []string{result.State.Messages[2].ToolName, result.State.Messages[3].ToolName}; !reflect.DeepEqual(got, []string{"slow", "fast"}) {
		t.Fatalf("persistence order = %v", got)
	}
}

func TestLengthStopRejectsToolWithoutExecution(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "danger", ArgumentsDelta: `{"path":"/"}`}}, StopReason: StopReasonLength}},
		{{TextDelta: "retried safely", StopReason: StopReasonStop}},
	}}
	var executions atomic.Int32
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "danger"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
		executions.Add(1)
		return TextToolResult("bad"), nil
	}}
	agent, err := New(Config{Model: model, Tools: []Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "run")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if executions.Load() != 0 {
		t.Fatalf("tool executed %d times", executions.Load())
	}
	toolMessage := result.State.Messages[2]
	if toolMessage.Role != RoleTool || toolMessage.Text() == "" {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestToolCancellationEndsRunAsAborted(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{{{
		ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "wait", ArgumentsDelta: `{}`}},
		StopReason:     StopReasonToolUse,
	}}}}
	started := make(chan struct{})
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "wait"}, ExecuteFunc: func(ctx context.Context, _ json.RawMessage, _ ToolUpdateSink) (ToolResult, error) {
		close(started)
		<-ctx.Done()
		return ToolResult{}, ctx.Err()
	}}
	agent, err := New(Config{Model: model, Tools: []Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var eventsMu sync.Mutex
	var events []Event
	done := make(chan struct{})
	var result Result
	var runErr error
	go func() {
		result, runErr = agent.Prompt(ctx, State{}, []Message{TextMessage(RoleUser, "wait")}, func(_ context.Context, event Event) error {
			eventsMu.Lock()
			events = append(events, event)
			eventsMu.Unlock()
			return nil
		})
		close(done)
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("tool did not start")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("run did not stop")
	}
	if !errors.Is(runErr, context.Canceled) || result.StopReason != StopReasonAborted {
		t.Fatalf("result = %+v, err = %v", result, runErr)
	}
	if len(result.State.Messages) != 3 {
		t.Fatalf("interrupted tool history has %d messages, want 3: %+v", len(result.State.Messages), result.State.Messages)
	}
	toolMessage := result.State.Messages[2]
	if toolMessage.Role != RoleTool || toolMessage.ToolCallID != "call-1" || !toolMessage.IsError || toolMessage.Text() == "" {
		t.Fatalf("interrupted tool result = %+v", toolMessage)
	}
	eventsMu.Lock()
	defer eventsMu.Unlock()
	if len(events) == 0 || events[len(events)-1].Type != EventAgentEnd {
		t.Fatalf("last event = %+v", events[len(events)-1])
	}
}

func TestEventStreamResultDoesNotDependOnConsumerDrain(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{{{TextDelta: "done", StopReason: StopReasonStop}}}}
	agent, err := New(Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	stream := agent.Stream(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")})
	result, err := stream.Result()
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != StopReasonStop {
		t.Fatalf("result = %+v", result)
	}
	var types []EventType
	for {
		event, ok := stream.Next()
		if !ok {
			break
		}
		types = append(types, event.Type)
	}
	if len(types) == 0 || types[0] != EventAgentStart || types[len(types)-1] != EventAgentEnd {
		t.Fatalf("event types = %v", types)
	}
}

func TestHooksCanRewriteArgumentsAndTerminateToolLoop(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{{{
		ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "finish", ArgumentsDelta: `{"value":"old"}`}},
		StopReason:     StopReasonToolUse,
	}}}}
	tool := FuncTool{
		ToolDefinition: ToolDefinition{Name: "finish"},
		ExecuteFunc: func(_ context.Context, arguments json.RawMessage, _ ToolUpdateSink) (ToolResult, error) {
			if string(arguments) != `{"value":"new"}` {
				t.Fatalf("arguments = %s", arguments)
			}
			return TextToolResult("finished"), nil
		},
	}
	agent, err := New(Config{
		Model: model,
		Tools: []Tool{tool},
		Hooks: Hooks{
			BeforeToolCall: func(context.Context, ToolCallContext) (ToolCallDecision, error) {
				return ToolCallDecision{Arguments: json.RawMessage(`{"value":"new"}`)}, nil
			},
			AfterToolCall: func(_ context.Context, _ ToolCallContext, result *ToolResult) error {
				result.Terminate = true
				return nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "finish")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != StopReasonTerminated || result.Turns != 1 {
		t.Fatalf("result = %+v", result)
	}
}

func TestModelStartErrorProducesClosedAssistantMessage(t *testing.T) {
	model := &fakeModel{}
	agent, err := New(Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	var events []Event
	result, runErr := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, func(_ context.Context, event Event) error {
		events = append(events, event)
		return nil
	})
	if runErr == nil || result.StopReason != StopReasonError {
		t.Fatalf("result = %+v, err = %v", result, runErr)
	}
	if len(result.State.Messages) != 2 {
		t.Fatalf("messages = %+v", result.State.Messages)
	}
	assistant := result.State.Messages[1]
	if assistant.Role != RoleAssistant || !assistant.IsError || assistant.Error == "" {
		t.Fatalf("assistant = %+v", assistant)
	}
	if got := eventTypes(events); len(got) < 3 || got[len(got)-3] != EventMessageStart || got[len(got)-2] != EventMessageEnd || got[len(got)-1] != EventAgentEnd {
		t.Fatalf("events = %v", got)
	}
}

func TestMaxTurnsStopsAnEndlessToolLoop(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{{{
		ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "again", ArgumentsDelta: `{}`}},
		StopReason:     StopReasonToolUse,
	}}}}
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "again"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
		return TextToolResult("continue"), nil
	}}
	agent, err := New(Config{Model: model, Tools: []Tool{tool}, MaxTurns: 1})
	if err != nil {
		t.Fatal(err)
	}
	result, runErr := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "loop")}, nil)
	if !errors.Is(runErr, ErrMaxTurns) || result.StopReason != StopReasonMaxTurns || result.Turns != 1 {
		t.Fatalf("result = %+v, err = %v", result, runErr)
	}
}

func TestSequentialToolModeNeverOverlapsExecutions(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{
			{Index: 0, ID: "call-1", Name: "first", ArgumentsDelta: `{}`},
			{Index: 1, ID: "call-2", Name: "second", ArgumentsDelta: `{}`},
		}, StopReason: StopReasonToolUse}},
		{{TextDelta: "done", StopReason: StopReasonStop}},
	}}
	var active atomic.Int32
	var overlapped atomic.Bool
	newTool := func(name string) Tool {
		return FuncTool{ToolDefinition: ToolDefinition{Name: name}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
			if active.Add(1) != 1 {
				overlapped.Store(true)
			}
			time.Sleep(5 * time.Millisecond)
			active.Add(-1)
			return TextToolResult(name), nil
		}}
	}
	agent, err := New(Config{
		Model:         model,
		Tools:         []Tool{newTool("first"), newTool("second")},
		ToolExecution: ToolExecutionSequential,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "run")}, nil); err != nil {
		t.Fatal(err)
	}
	if overlapped.Load() {
		t.Fatal("sequential tools overlapped")
	}
}

func eventTypes(events []Event) []EventType {
	result := make([]EventType, len(events))
	for i := range events {
		result[i] = events[i].Type
	}
	return result
}
