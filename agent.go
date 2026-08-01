package agentcore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	validator "github.com/santhosh-tekuri/jsonschema/v6"
)

const defaultMaxTurns = 64

var messageSequence atomic.Uint64

// Agent owns immutable loop configuration and may be reused concurrently.
type Agent struct {
	config     Config
	tools      map[string]Tool
	validators map[string]*validator.Schema
}

// New validates config and creates an Agent.
func New(config Config) (*Agent, error) {
	if config.Model == nil {
		return nil, ErrModelRequired
	}
	if config.MaxTurns <= 0 {
		config.MaxTurns = defaultMaxTurns
	}
	if config.ToolExecution == "" {
		config.ToolExecution = ToolExecutionParallel
	}
	if config.ToolExecution != ToolExecutionParallel && config.ToolExecution != ToolExecutionSequential {
		return nil, fmt.Errorf("agentcore: unsupported tool execution mode %q", config.ToolExecution)
	}

	tools := make(map[string]Tool, len(config.Tools))
	validators := make(map[string]*validator.Schema, len(config.Tools))
	for index, tool := range config.Tools {
		if tool == nil {
			return nil, errors.New("agentcore: nil tool")
		}
		definition := tool.Definition()
		if definition.Name == "" {
			return nil, errors.New("agentcore: tool name is required")
		}
		if _, exists := tools[definition.Name]; exists {
			return nil, fmt.Errorf("agentcore: duplicate tool %q", definition.Name)
		}
		tools[definition.Name] = tool
		if len(definition.Parameters) > 0 {
			compiled, err := compileToolSchema(index, definition)
			if err != nil {
				return nil, err
			}
			validators[definition.Name] = compiled
		}
	}

	config.Tools = append([]Tool(nil), config.Tools...)
	config.Interceptors = append([]Interceptor(nil), config.Interceptors...)
	for index, interceptor := range config.Interceptors {
		if interceptor == nil {
			return nil, fmt.Errorf("agentcore: nil interceptor at index %d", index)
		}
	}
	config.ModelOptions = cloneOptions(config.ModelOptions)
	config.ModelRetry = normalizeRetryPolicy(config.ModelRetry)
	config.ToolPolicy = normalizeToolPolicy(config.ToolPolicy)
	return &Agent{config: config, tools: tools, validators: validators}, nil
}

func compileToolSchema(index int, definition ToolDefinition) (*validator.Schema, error) {
	if !json.Valid(definition.Parameters) {
		return nil, fmt.Errorf("agentcore: invalid JSON schema for tool %q", definition.Name)
	}
	decoder := json.NewDecoder(bytes.NewReader(definition.Parameters))
	decoder.UseNumber()
	var document any
	if err := decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("agentcore: invalid JSON schema for tool %q: %w", definition.Name, err)
	}
	compiler := validator.NewCompiler()
	location := fmt.Sprintf("urn:agentcore:tool:%d", index)
	if err := compiler.AddResource(location, document); err != nil {
		return nil, fmt.Errorf("agentcore: add JSON schema for tool %q: %w", definition.Name, err)
	}
	compiled, err := compiler.Compile(location)
	if err != nil {
		return nil, fmt.Errorf("agentcore: compile JSON schema for tool %q: %w", definition.Name, err)
	}
	return compiled, nil
}

// Prompt appends prompts to state and runs model/tool turns until the model
// stops, a tool terminates the loop, the limit is reached, or ctx is canceled.
// Events are delivered synchronously through sink. A nil sink discards events.
func (a *Agent) Prompt(ctx context.Context, state State, prompts []Message, sink EventSink) (Result, error) {
	return a.prompt(ctx, state, prompts, sink, nil)
}

func (a *Agent) prompt(ctx context.Context, state State, prompts []Message, sink EventSink, queue messageQueue) (Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	emit := newEmitter(ctx, sink)
	current := State{Messages: cloneMessages(state.Messages)}
	newMessages := make([]Message, 0, len(prompts)+4)

	if err := emit.send(Event{Type: EventAgentStart}); err != nil {
		return resultFrom(current, newMessages, 0, StopReasonError), err
	}
	if err := emit.send(Event{Type: EventTurnStart, Turn: 1}); err != nil {
		return resultFrom(current, newMessages, 0, StopReasonError), err
	}
	for _, prompt := range prompts {
		prompt = cloneMessage(prompt)
		ensureMessageID(&prompt)
		current.Messages = append(current.Messages, prompt)
		newMessages = append(newMessages, prompt)
		copy := cloneMessage(prompt)
		if err := emit.send(Event{Type: EventMessageStart, Turn: 1, Message: &copy}); err != nil {
			return resultFrom(current, newMessages, 0, StopReasonError), err
		}
		if err := emit.send(Event{Type: EventMessageEnd, Turn: 1, Message: &copy}); err != nil {
			return resultFrom(current, newMessages, 0, StopReasonError), err
		}
	}

	return a.run(ctx, current, newMessages, emit, true, queue)
}

// Continue resumes a state without adding a prompt. The final existing message
// must not be an assistant message because providers require user or tool input
// before another assistant response.
func (a *Agent) Continue(ctx context.Context, state State, sink EventSink) (Result, error) {
	if len(state.Messages) == 0 {
		return Result{}, errors.New("agentcore: cannot continue empty state")
	}
	if state.Messages[len(state.Messages)-1].Role == RoleAssistant {
		return Result{}, errors.New("agentcore: cannot continue from assistant message")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	emit := newEmitter(ctx, sink)
	current := State{Messages: cloneMessages(state.Messages)}
	if err := emit.send(Event{Type: EventAgentStart}); err != nil {
		return resultFrom(current, nil, 0, StopReasonError), err
	}
	if err := emit.send(Event{Type: EventTurnStart, Turn: 1}); err != nil {
		return resultFrom(current, nil, 0, StopReasonError), err
	}
	return a.run(ctx, current, nil, emit, true, nil)
}

func (a *Agent) run(ctx context.Context, state State, newMessages []Message, emit *eventEmitter, turnAlreadyStarted bool, queue messageQueue) (Result, error) {
	active := a
	var pendingMessages []Message
	for turn := 1; turn <= a.config.MaxTurns; turn++ {
		if err := ctx.Err(); err != nil {
			return a.finish(state, newMessages, turn-1, StopReasonAborted, err, emit)
		}
		if !turnAlreadyStarted {
			if err := emit.send(Event{Type: EventTurnStart, Turn: turn}); err != nil {
				return resultFrom(state, newMessages, turn-1, StopReasonError), err
			}
		}
		turnAlreadyStarted = false
		for _, pending := range pendingMessages {
			pending = cloneMessage(pending)
			ensureMessageID(&pending)
			state.Messages = append(state.Messages, pending)
			newMessages = append(newMessages, pending)
			copy := cloneMessage(pending)
			if err := emit.send(Event{Type: EventMessageStart, Turn: turn, Message: &copy}); err != nil {
				return resultFrom(state, newMessages, turn-1, StopReasonError), err
			}
			if err := emit.send(Event{Type: EventMessageEnd, Turn: turn, Message: &copy}); err != nil {
				return resultFrom(state, newMessages, turn-1, StopReasonError), err
			}
		}
		pendingMessages = nil

		assistant, err := active.streamAssistant(ctx, state, turn, emit)
		state.Messages = append(state.Messages, assistant)
		newMessages = append(newMessages, assistant)
		if err != nil {
			stop := StopReasonError
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				stop = StopReasonAborted
			}
			return a.finish(state, newMessages, turn, stop, err, emit)
		}

		calls := assistant.ToolCalls()
		toolMessages := make([]Message, 0, len(calls))
		terminate := false
		if len(calls) > 0 {
			var outcomes []toolOutcome
			if assistant.StopReason == StopReasonLength {
				outcomes, err = active.rejectTruncatedCalls(ctx, calls, turn, emit)
			} else {
				outcomes, err = active.executeToolCalls(ctx, &state, calls, turn, emit)
			}
			if err != nil {
				stop := StopReasonError
				if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
					stop = StopReasonAborted
				}
				return a.finish(state, newMessages, turn, stop, err, emit)
			}
			terminate = outcomesTerminate(outcomes)
			for _, outcome := range outcomes {
				message := toolResultMessage(outcome.call, outcome.result)
				state.Messages = append(state.Messages, message)
				newMessages = append(newMessages, message)
				toolMessages = append(toolMessages, message)
				copy := cloneMessage(message)
				if err := emit.send(Event{Type: EventMessageStart, Turn: turn, Message: &copy}); err != nil {
					return resultFrom(state, newMessages, turn, StopReasonError), err
				}
				if err := emit.send(Event{Type: EventMessageEnd, Turn: turn, Message: &copy}); err != nil {
					return resultFrom(state, newMessages, turn, StopReasonError), err
				}
			}
		}

		turnContext := TurnContext{
			Turn:        turn,
			Message:     cloneMessage(assistant),
			ToolResults: cloneMessages(toolMessages),
			State:       &state,
			Usage:       aggregateUsage(newMessages),
		}
		if err := emit.send(Event{Type: EventTurnEnd, Turn: turn, Message: &turnContext.Message, ToolResults: cloneMessages(toolMessages)}); err != nil {
			return resultFrom(state, newMessages, turn, StopReasonError), err
		}
		if err := active.prepareNextTurn(ctx, &turnContext); err != nil {
			return a.finish(state, newMessages, turn, StopReasonError, err, emit)
		}
		if turnContext.Next != nil {
			var err error
			active, err = active.withNextTurn(*turnContext.Next)
			if err != nil {
				return a.finish(state, newMessages, turn, StopReasonError, err, emit)
			}
		}
		if active.shouldStop(ctx, turnContext) {
			return a.finish(state, newMessages, turn, StopReasonStop, nil, emit)
		}
		if queue != nil {
			pendingMessages = queue.takeSteering()
			if len(pendingMessages) > 0 {
				continue
			}
		}
		if terminate {
			return a.finish(state, newMessages, turn, StopReasonTerminated, nil, emit)
		}
		if len(calls) == 0 {
			if queue != nil {
				pendingMessages = queue.takeFollowUp()
				if len(pendingMessages) > 0 {
					continue
				}
			}
			stop := assistant.StopReason
			if stop == "" {
				stop = StopReasonStop
			}
			return a.finish(state, newMessages, turn, stop, nil, emit)
		}
	}

	return a.finish(state, newMessages, a.config.MaxTurns, StopReasonMaxTurns, ErrMaxTurns, emit)
}

func (a *Agent) withNextTurn(next NextTurnConfig) (*Agent, error) {
	config := a.config
	if next.Model != nil {
		config.Model = next.Model
	}
	if next.SystemPrompt != nil {
		config.SystemPrompt = *next.SystemPrompt
	}
	if next.Tools != nil {
		config.Tools = append([]Tool(nil), (*next.Tools)...)
	}
	if next.ToolExecution != "" {
		config.ToolExecution = next.ToolExecution
	}
	if next.ModelOptions != nil {
		config.ModelOptions = cloneOptions(next.ModelOptions)
	}
	updated, err := New(config)
	if err != nil {
		return nil, fmt.Errorf("agentcore: prepare next turn: %w", err)
	}
	return updated, nil
}

func (a *Agent) streamAssistant(ctx context.Context, state State, turn int, emit *eventEmitter) (Message, error) {
	policy := a.config.ModelRetry
	var lastMessage Message
	var lastErr error
	retried := false
	lastAttempt := 0
	for attempt := 1; attempt <= policy.MaxAttempts; attempt++ {
		lastAttempt = attempt
		message, err := a.streamAssistantAttempt(ctx, state, turn, emit)
		if err == nil {
			if retried {
				if emitErr := emit.send(Event{Type: EventAutoRetryEnd, Turn: turn, Attempt: attempt, MaxAttempts: policy.MaxAttempts, Success: true}); emitErr != nil {
					return message, emitErr
				}
			}
			return message, nil
		}
		lastMessage, lastErr = message, err
		if attempt == policy.MaxAttempts || !shouldRetryModelError(ctx, err, policy) || emit.failure() != nil {
			break
		}
		retried = true
		delay := retryDelay(policy, attempt)
		if err := emit.send(Event{
			Type: EventAutoRetryStart, Turn: turn, Attempt: attempt + 1,
			MaxAttempts: policy.MaxAttempts, Delay: delay, Error: lastErr.Error(),
		}); err != nil {
			return lastMessage, err
		}
		if delay > 0 {
			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				timer.Stop()
				lastErr = ctx.Err()
				attempt = policy.MaxAttempts
			case <-timer.C:
			}
		}
	}
	if retried {
		_ = emit.send(Event{
			Type: EventAutoRetryEnd, Turn: turn, Attempt: lastAttempt,
			MaxAttempts: policy.MaxAttempts, Success: false, IsError: true, Error: lastErr.Error(),
		})
	}
	return lastMessage, lastErr
}

type modelCallError struct {
	err error
}

func (e modelCallError) Error() string { return e.err.Error() }
func (e modelCallError) Unwrap() error { return e.err }

func shouldRetryModelError(ctx context.Context, err error, policy RetryPolicy) bool {
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var modelErr modelCallError
	if !errors.As(err, &modelErr) {
		return false
	}
	return policy.ShouldRetry == nil || policy.ShouldRetry(err)
}

func (a *Agent) streamAssistantAttempt(ctx context.Context, state State, turn int, emit *eventEmitter) (Message, error) {
	assistant := Message{ID: nextMessageID("assistant"), Role: RoleAssistant}
	fail := func(runErr error, started bool) (Message, error) {
		assistant.StopReason = StopReasonError
		assistant.Error = runErr.Error()
		assistant.IsError = true
		copy := cloneMessage(assistant)
		if !started {
			_ = emit.send(Event{Type: EventMessageStart, Turn: turn, Message: &copy})
		}
		_ = emit.send(Event{Type: EventMessageEnd, Turn: turn, Message: &copy, IsError: true, Error: runErr.Error()})
		return assistant, runErr
	}
	messages := cloneMessages(state.Messages)
	if a.config.TransformContext != nil {
		var err error
		messages, err = a.config.TransformContext(ctx, messages)
		if err != nil {
			return fail(fmt.Errorf("transform context: %w", err), false)
		}
	}
	messages, err := a.applyContextPolicy(ctx, messages, turn, emit)
	if err != nil {
		return fail(err, false)
	}
	request := ModelRequest{
		SystemPrompt: a.config.SystemPrompt,
		Messages:     messages,
		Tools:        make([]ToolDefinition, 0, len(a.config.Tools)),
		Options:      cloneOptions(a.config.ModelOptions),
	}
	for _, tool := range a.config.Tools {
		request.Tools = append(request.Tools, cloneToolDefinition(tool.Definition()))
	}
	if err := a.beforeModelCall(ctx, &request); err != nil {
		return fail(fmt.Errorf("before model call: %w", err), false)
	}

	stream, err := startModelStream(ctx, a.config.Model, request)
	if err != nil {
		return fail(err, false)
	}
	if stream == nil {
		return fail(modelCallError{err: errors.New("start model stream: model returned a nil stream")}, false)
	}
	defer closeModelStream(stream)

	startCopy := cloneMessage(assistant)
	if err := emit.send(Event{Type: EventMessageStart, Turn: turn, Message: &startCopy}); err != nil {
		return assistant, err
	}
	accumulator := responseAccumulator{message: &assistant, toolBlockIndexes: map[int]int{}}
	for {
		chunk, recvErr := receiveModelChunk(stream)
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			return fail(recvErr, true)
		}
		accumulator.add(chunk)
		updateCopy := cloneMessage(assistant)
		chunkCopy := cloneModelChunk(chunk)
		if err := emit.send(Event{Type: EventMessageUpdate, Turn: turn, Message: &updateCopy, Delta: &chunkCopy}); err != nil {
			return assistant, err
		}
	}
	if assistant.StopReason == "" {
		if len(assistant.ToolCalls()) > 0 {
			assistant.StopReason = StopReasonToolUse
		} else {
			assistant.StopReason = StopReasonStop
		}
	}
	if a.config.Hooks.AfterModelCall != nil || len(a.config.Interceptors) > 0 {
		beforeHook := cloneMessage(assistant)
		if err := a.afterModelCall(ctx, &assistant); err != nil {
			return fail(fmt.Errorf("after model call: %w", err), true)
		}
		// ProviderData represents the exact provider message accumulated above.
		// If a hook changes model-visible fields without replacing ProviderData,
		// retaining it would silently discard those changes on the next request.
		if !sameModelVisibleMessage(beforeHook, assistant) && reflect.DeepEqual(beforeHook.ProviderData, assistant.ProviderData) {
			assistant.ProviderData = nil
		}
	}
	if normalizeAssistantToolCalls(&assistant) {
		// The preserved provider message does not contain the generated IDs or
		// contains duplicate IDs, so future adapters must use the public message.
		assistant.ProviderData = nil
	}
	endCopy := cloneMessage(assistant)
	if err := emit.send(Event{Type: EventMessageEnd, Turn: turn, Message: &endCopy}); err != nil {
		return assistant, err
	}
	return assistant, nil
}

func startModelStream(ctx context.Context, target Model, request ModelRequest) (stream ModelStream, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			stream = nil
			err = modelCallError{err: fmt.Errorf("start model stream: model panicked: %v", recovered)}
		}
	}()
	stream, err = target.Stream(ctx, request)
	if err != nil {
		return nil, modelCallError{err: fmt.Errorf("start model stream: %w", err)}
	}
	return stream, nil
}

func receiveModelChunk(stream ModelStream) (chunk ModelChunk, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			chunk = ModelChunk{}
			err = modelCallError{err: fmt.Errorf("receive model stream: model panicked: %v", recovered)}
		}
	}()
	chunk, err = stream.Recv()
	if err != nil && !errors.Is(err, io.EOF) {
		return ModelChunk{}, modelCallError{err: fmt.Errorf("receive model stream: %w", err)}
	}
	return chunk, err
}

func closeModelStream(stream ModelStream) {
	defer func() { _ = recover() }()
	_ = stream.Close()
}

func sameModelVisibleMessage(left, right Message) bool {
	return left.Role == right.Role &&
		left.ToolCallID == right.ToolCallID &&
		left.ToolName == right.ToolName &&
		reflect.DeepEqual(left.Content, right.Content)
}

func normalizeAssistantToolCalls(message *Message) bool {
	providerInvalid := false
	seen := make(map[string]struct{})
	for index := range message.Content {
		block := &message.Content[index]
		if block.Type != ContentToolCall || block.ToolCall == nil {
			continue
		}
		if block.ToolCall.ID == "" {
			block.ToolCall.ID = nextMessageID("call")
			providerInvalid = true
		}
		if _, duplicate := seen[block.ToolCall.ID]; duplicate {
			block.ToolCall.ID = nextMessageID("call")
			providerInvalid = true
		}
		seen[block.ToolCall.ID] = struct{}{}
		if len(block.ToolCall.Arguments) == 0 {
			block.ToolCall.Arguments = json.RawMessage("{}")
		}
	}
	return providerInvalid
}

func (a *Agent) finish(state State, newMessages []Message, turns int, reason StopReason, runErr error, emit *eventEmitter) (Result, error) {
	result := resultFrom(state, newMessages, turns, reason)
	event := Event{Type: EventAgentEnd, Turn: turns, Messages: cloneMessages(newMessages)}
	if runErr != nil {
		event.IsError = true
		event.Error = runErr.Error()
	}
	if err := emit.send(event); err != nil && runErr == nil {
		return result, err
	}
	return result, runErr
}

type responseAccumulator struct {
	message          *Message
	toolBlockIndexes map[int]int
}

func (a *responseAccumulator) add(chunk ModelChunk) {
	if chunk.TextDelta != "" {
		a.appendText(ContentText, chunk.TextDelta)
	}
	if chunk.ThinkingDelta != "" {
		a.appendText(ContentThinking, chunk.ThinkingDelta)
	}
	for _, block := range chunk.ContentDeltas {
		if block.Type == ContentText || block.Type == ContentThinking {
			a.appendText(block.Type, block.Text)
		} else {
			a.message.Content = append(a.message.Content, cloneContent([]ContentBlock{block})[0])
		}
	}
	for _, delta := range chunk.ToolCallDeltas {
		blockIndex, exists := a.toolBlockIndexes[delta.Index]
		if !exists {
			call := ToolCall{ID: delta.ID, Name: delta.Name}
			a.message.Content = append(a.message.Content, ContentBlock{Type: ContentToolCall, ToolCall: &call})
			blockIndex = len(a.message.Content) - 1
			a.toolBlockIndexes[delta.Index] = blockIndex
		}
		call := a.message.Content[blockIndex].ToolCall
		if delta.ID != "" {
			call.ID = delta.ID
		}
		if delta.Name != "" {
			call.Name = delta.Name
		}
		call.Arguments = append(call.Arguments, delta.ArgumentsDelta...)
	}
	if chunk.StopReason != "" {
		a.message.StopReason = chunk.StopReason
	}
	if chunk.Usage != nil {
		a.message.Usage = *chunk.Usage
	}
	if chunk.ProviderData != nil {
		a.message.ProviderData = chunk.ProviderData
	}
}

func (a *responseAccumulator) appendText(kind ContentType, delta string) {
	if len(a.message.Content) > 0 && a.message.Content[len(a.message.Content)-1].Type == kind {
		a.message.Content[len(a.message.Content)-1].Text += delta
		return
	}
	a.message.Content = append(a.message.Content, ContentBlock{Type: kind, Text: delta})
}

type eventEmitter struct {
	ctx  context.Context
	sink EventSink
	mu   sync.Mutex
	err  error
}

func newEmitter(ctx context.Context, sink EventSink) *eventEmitter {
	if sink == nil {
		sink = func(context.Context, Event) error { return nil }
	}
	return &eventEmitter{ctx: ctx, sink: sink}
}

func (e *eventEmitter) send(event Event) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.err != nil {
		return e.err
	}
	if err := e.sink(e.ctx, event); err != nil {
		e.err = err
		return err
	}
	return nil
}

func (e *eventEmitter) failure() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.err
}

func resultFrom(state State, newMessages []Message, turns int, reason StopReason) Result {
	return Result{
		State:       State{Messages: cloneMessages(state.Messages)},
		NewMessages: cloneMessages(newMessages),
		Turns:       turns,
		StopReason:  reason,
		Usage:       aggregateUsage(newMessages),
	}
}

func aggregateUsage(messages []Message) Usage {
	var usage Usage
	for _, message := range messages {
		if message.Role == RoleAssistant {
			usage = usage.Add(message.Usage)
		}
	}
	return usage
}

func ensureMessageID(message *Message) {
	if message.ID == "" {
		message.ID = nextMessageID(string(message.Role))
	}
}

func nextMessageID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, messageSequence.Add(1))
}

func cloneMessages(messages []Message) []Message {
	result := make([]Message, len(messages))
	for i := range messages {
		result[i] = cloneMessage(messages[i])
	}
	return result
}

func cloneMessage(message Message) Message {
	content := message.Content
	message.Content = make([]ContentBlock, len(content))
	for i, block := range content {
		message.Content[i] = block
		message.Content[i].Data = append([]byte(nil), block.Data...)
		if block.ToolCall != nil {
			call := cloneToolCall(*block.ToolCall)
			message.Content[i].ToolCall = &call
		}
	}
	message.ProviderData = cloneProviderData(message.ProviderData)
	return message
}

func cloneToolDefinition(definition ToolDefinition) ToolDefinition {
	definition.Parameters = append(json.RawMessage(nil), definition.Parameters...)
	return definition
}

func cloneModelChunk(chunk ModelChunk) ModelChunk {
	chunk.ContentDeltas = cloneContent(chunk.ContentDeltas)
	chunk.ToolCallDeltas = append([]ToolCallDelta(nil), chunk.ToolCallDeltas...)
	if chunk.Usage != nil {
		usage := *chunk.Usage
		chunk.Usage = &usage
	}
	chunk.ProviderData = cloneProviderData(chunk.ProviderData)
	return chunk
}

func cloneProviderData(providerData *ProviderData) *ProviderData {
	if providerData == nil {
		return nil
	}
	copy := *providerData
	copy.Data = append(json.RawMessage(nil), providerData.Data...)
	return &copy
}

func cloneOptions(options map[string]any) map[string]any {
	if options == nil {
		return nil
	}
	copy := make(map[string]any, len(options))
	for key, value := range options {
		copy[key] = value
	}
	return copy
}
