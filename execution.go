package agentcore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"
)

type toolOutcome struct {
	call        ToolCall
	result      ToolResult
	attempt     int
	maxAttempts int
}

type preparedToolCall struct {
	call        ToolCall
	tool        Tool
	callContext ToolCallContext
	immediate   *ToolResult
}

func (a *Agent) rejectTruncatedCalls(ctx context.Context, calls []ToolCall, turn int, emit *eventEmitter) ([]toolOutcome, error) {
	outcomes := make([]toolOutcome, 0, len(calls))
	for _, call := range calls {
		if err := emit.send(toolStartEvent(turn, call)); err != nil {
			return nil, err
		}
		result := errorToolResult(fmt.Sprintf("tool call %q was not executed because model output hit its token limit and arguments may be truncated", call.Name))
		outcome := toolOutcome{call: call, result: result}
		if err := emit.send(toolEndEvent(turn, outcome)); err != nil {
			return nil, err
		}
		outcomes = append(outcomes, outcome)
	}
	return outcomes, ctx.Err()
}

func (a *Agent) executeToolCalls(ctx context.Context, state *State, calls []ToolCall, turn int, emit *eventEmitter) ([]toolOutcome, error) {
	sequential := a.config.ToolExecution == ToolExecutionSequential
	if !sequential {
		for _, call := range calls {
			if tool := a.tools[call.Name]; tool != nil {
				if provider, ok := tool.(ToolExecutionModeProvider); ok && provider.ExecutionMode() == ToolExecutionSequential {
					sequential = true
					break
				}
			}
		}
	}
	if sequential {
		return a.executeSequential(ctx, state, calls, turn, emit)
	}
	return a.executeParallel(ctx, state, calls, turn, emit)
}

func (a *Agent) executeSequential(ctx context.Context, state *State, calls []ToolCall, turn int, emit *eventEmitter) ([]toolOutcome, error) {
	outcomes := make([]toolOutcome, 0, len(calls))
	for _, call := range calls {
		if err := emit.send(toolStartEvent(turn, call)); err != nil {
			return outcomes, err
		}
		prepared := a.prepareToolCall(ctx, state, call, turn)
		outcome := a.executePrepared(ctx, prepared, turn, emit)
		outcome = a.finalizeToolCall(ctx, prepared, outcome)
		if err := emit.send(toolEndEvent(turn, outcome)); err != nil {
			return outcomes, err
		}
		outcomes = append(outcomes, outcome)
		if err := ctx.Err(); err != nil {
			return outcomes, err
		}
	}
	return outcomes, nil
}

func (a *Agent) executeParallel(ctx context.Context, state *State, calls []ToolCall, turn int, emit *eventEmitter) ([]toolOutcome, error) {
	type indexedOutcome struct {
		index    int
		prepared preparedToolCall
		outcome  toolOutcome
	}
	toolCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	completed := make(chan indexedOutcome, len(calls))
	outcomes := make([]toolOutcome, len(calls))
	pending := make([]indexedOutcome, 0, len(calls))
	for index, call := range calls {
		if err := emit.send(toolStartEvent(turn, call)); err != nil {
			return nil, err
		}
		prepared := a.prepareToolCall(toolCtx, state, call, turn)
		if prepared.immediate != nil {
			outcome := toolOutcome{call: prepared.call, result: cloneToolResult(*prepared.immediate)}
			outcome = a.finalizeToolCall(toolCtx, prepared, outcome)
			outcomes[index] = outcome
			if err := emit.send(toolEndEvent(turn, outcome)); err != nil {
				return nil, err
			}
			continue
		}
		pending = append(pending, indexedOutcome{index: index, prepared: prepared})
	}
	var semaphore chan struct{}
	if a.config.MaxToolConcurrency > 0 && a.config.MaxToolConcurrency < len(pending) {
		semaphore = make(chan struct{}, a.config.MaxToolConcurrency)
	}
	for _, entry := range pending {
		go func() {
			if semaphore != nil {
				select {
				case semaphore <- struct{}{}:
					defer func() { <-semaphore }()
				case <-toolCtx.Done():
					completed <- indexedOutcome{
						index: entry.index, prepared: entry.prepared,
						outcome: toolOutcome{call: entry.prepared.call, result: errorToolResult(toolCtx.Err().Error())},
					}
					return
				}
			}
			completed <- indexedOutcome{
				index:    entry.index,
				prepared: entry.prepared,
				outcome:  a.executePrepared(toolCtx, entry.prepared, turn, emit),
			}
		}()
	}
	for range len(pending) {
		completedOutcome := <-completed
		outcome := a.finalizeToolCall(toolCtx, completedOutcome.prepared, completedOutcome.outcome)
		outcomes[completedOutcome.index] = outcome
		if err := emit.send(toolEndEvent(turn, outcome)); err != nil {
			cancel()
			return outcomes, err
		}
	}
	if err := ctx.Err(); err != nil {
		return outcomes, err
	}
	return outcomes, nil
}

func (a *Agent) prepareToolCall(ctx context.Context, state *State, call ToolCall, turn int) preparedToolCall {
	call = cloneToolCall(call)
	prepared := preparedToolCall{call: call}
	if len(call.Arguments) == 0 {
		call.Arguments = json.RawMessage("{}")
	}
	callContext := ToolCallContext{Turn: turn, Call: cloneToolCall(call), State: state}
	callContext, immediate := a.beforeToolCall(ctx, callContext)
	call = cloneToolCall(callContext.Call)
	prepared.call = call
	prepared.callContext = callContext
	if immediate != nil {
		prepared.immediate = immediate
		return prepared
	}
	tool := a.tools[call.Name]
	if tool == nil {
		result := errorToolResult(fmt.Sprintf("tool %q not found", call.Name))
		prepared.immediate = &result
		return prepared
	}
	prepared.tool = tool
	if err := a.validateToolArguments(tool, call); err != nil {
		result := errorToolResult(err.Error())
		prepared.immediate = &result
		return prepared
	}
	return prepared
}

func (a *Agent) validateToolArguments(tool Tool, call ToolCall) error {
	if !json.Valid(call.Arguments) {
		return fmt.Errorf("invalid JSON arguments for tool %q", call.Name)
	}
	if schemaValidator := a.validators[call.Name]; schemaValidator != nil {
		decoder := json.NewDecoder(bytes.NewReader(call.Arguments))
		decoder.UseNumber()
		var value any
		if err := decoder.Decode(&value); err != nil {
			return fmt.Errorf("invalid arguments for tool %q: %v", call.Name, err)
		}
		if err := schemaValidator.Validate(value); err != nil {
			return fmt.Errorf("arguments do not match schema for tool %q: %v", call.Name, err)
		}
	}
	if validator, ok := tool.(ToolValidator); ok {
		if err := validator.Validate(call.Arguments); err != nil {
			return fmt.Errorf("invalid arguments for tool %q: %v", call.Name, err)
		}
	}
	return nil
}

func (a *Agent) executePrepared(ctx context.Context, prepared preparedToolCall, turn int, emit *eventEmitter) toolOutcome {
	call := prepared.call
	if prepared.immediate != nil {
		return toolOutcome{call: call, result: cloneToolResult(*prepared.immediate)}
	}
	policy := a.config.ToolPolicy
	if provider, ok := prepared.tool.(ToolPolicyProvider); ok {
		policy = mergeToolPolicy(policy, provider.ToolPolicy())
	}
	update := func(attempt int) ToolUpdateSink {
		return func(partial ToolResult) error {
			partialCopy := cloneToolResult(partial)
			return emit.send(Event{
				Type:        EventToolExecutionUpdate,
				Turn:        turn,
				ToolCallID:  call.ID,
				ToolName:    call.Name,
				Arguments:   append(json.RawMessage(nil), call.Arguments...),
				ToolResult:  &partialCopy,
				IsError:     partial.IsError,
				Attempt:     attempt,
				MaxAttempts: policy.MaxAttempts,
			})
		}
	}
	var result ToolResult
	var err error
	attempt := 0
	for attempt = 1; attempt <= policy.MaxAttempts; attempt++ {
		invocation := ToolInvocation{Turn: turn, Attempt: attempt, MaxAttempts: policy.MaxAttempts, Call: cloneToolCall(call)}
		result, err = executeToolAttempt(ctx, prepared.tool, invocation, update(attempt), policy)
		if err == nil || ctx.Err() != nil || attempt == policy.MaxAttempts || (policy.ShouldRetry != nil && !policy.ShouldRetry(err)) {
			break
		}
		if policy.RetryDelay > 0 {
			timer := time.NewTimer(policy.RetryDelay)
			select {
			case <-ctx.Done():
				timer.Stop()
				err = ctx.Err()
			case <-timer.C:
			}
			if ctx.Err() != nil {
				break
			}
		}
	}
	if err != nil {
		result.IsError = true
		if result.Text() == "" {
			result.Content = []ContentBlock{{Type: ContentText, Text: err.Error()}}
		}
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		result.IsError = true
		if result.Text() == "" {
			result.Content = []ContentBlock{{Type: ContentText, Text: ctxErr.Error()}}
		}
	}
	return toolOutcome{call: call, result: cloneToolResult(result), attempt: attempt, maxAttempts: policy.MaxAttempts}
}

func executeToolAttempt(ctx context.Context, target Tool, invocation ToolInvocation, update ToolUpdateSink, policy ToolPolicy) (ToolResult, error) {
	call := func(callCtx context.Context) (result ToolResult, err error) {
		if !policy.DisablePanicRecovery {
			defer func() {
				if recovered := recover(); recovered != nil {
					err = fmt.Errorf("agentcore: tool panicked: %v", recovered)
				}
			}()
		}
		safeUpdate := func(result ToolResult) error {
			if err := callCtx.Err(); err != nil {
				return err
			}
			if update == nil {
				return nil
			}
			return update(result)
		}
		callCtx = withToolInvocation(callCtx, invocation)
		return target.Execute(callCtx, append(json.RawMessage(nil), invocation.Call.Arguments...), safeUpdate)
	}
	if policy.Timeout <= 0 {
		return call(ctx)
	}
	callCtx, cancel := context.WithTimeout(ctx, policy.Timeout)
	defer cancel()
	type response struct {
		result ToolResult
		err    error
	}
	completed := make(chan response, 1)
	go func() {
		result, err := call(callCtx)
		completed <- response{result: result, err: err}
	}()
	select {
	case response := <-completed:
		return response.result, response.err
	case <-callCtx.Done():
		return ToolResult{}, callCtx.Err()
	}
}

func (a *Agent) finalizeToolCall(ctx context.Context, prepared preparedToolCall, outcome toolOutcome) toolOutcome {
	a.afterToolCall(ctx, prepared.callContext, &outcome.result, prepared.immediate == nil, outcome.attempt)
	return outcome
}

func toolStartEvent(turn int, call ToolCall) Event {
	return Event{
		Type:       EventToolExecutionStart,
		Turn:       turn,
		ToolCallID: call.ID,
		ToolName:   call.Name,
		Arguments:  append(json.RawMessage(nil), call.Arguments...),
	}
}

func toolEndEvent(turn int, outcome toolOutcome) Event {
	resultCopy := cloneToolResult(outcome.result)
	return Event{
		Type:        EventToolExecutionEnd,
		Turn:        turn,
		ToolCallID:  outcome.call.ID,
		ToolName:    outcome.call.Name,
		Arguments:   append(json.RawMessage(nil), outcome.call.Arguments...),
		ToolResult:  &resultCopy,
		IsError:     outcome.result.IsError,
		Attempt:     outcome.attempt,
		MaxAttempts: outcome.maxAttempts,
	}
}

func toolResultMessage(call ToolCall, result ToolResult) Message {
	content := cloneContent(result.Content)
	if len(content) == 0 {
		content = []ContentBlock{{Type: ContentText, Text: ""}}
	}
	return Message{
		ID:         nextMessageID("tool"),
		Role:       RoleTool,
		Content:    content,
		ToolCallID: call.ID,
		ToolName:   call.Name,
		IsError:    result.IsError,
	}
}

func outcomesTerminate(outcomes []toolOutcome) bool {
	if len(outcomes) == 0 {
		return false
	}
	for _, outcome := range outcomes {
		if !outcome.result.Terminate {
			return false
		}
	}
	return true
}

func errorToolResult(message string) ToolResult {
	return ToolResult{Content: []ContentBlock{{Type: ContentText, Text: message}}, IsError: true}
}

func cloneToolResult(result ToolResult) ToolResult {
	result.Content = cloneContent(result.Content)
	return result
}

func cloneContent(content []ContentBlock) []ContentBlock {
	result := make([]ContentBlock, len(content))
	for i, block := range content {
		result[i] = block
		result[i].Data = append([]byte(nil), block.Data...)
		if block.ToolCall != nil {
			call := cloneToolCall(*block.ToolCall)
			result[i].ToolCall = &call
		}
	}
	return result
}
