package agentcore

import "context"

// Interceptor is a named lifecycle extension. Implement only the optional
// lifecycle interfaces needed by the extension. Interceptors run in
// registration order before an operation and reverse order after it.
type Interceptor interface {
	InterceptorName() string
}

type BeforeModelCallInterceptor interface {
	Interceptor
	BeforeModelCall(context.Context, *ModelRequest) error
}

type AfterModelCallInterceptor interface {
	Interceptor
	AfterModelCall(context.Context, *Message) error
}

type BeforeToolCallInterceptor interface {
	Interceptor
	BeforeToolCall(context.Context, ToolCallContext) (ToolCallDecision, error)
}

type AfterToolCallInterceptor interface {
	Interceptor
	AfterToolCall(context.Context, ToolCallContext, *ToolResult) error
}

type PrepareNextTurnInterceptor interface {
	Interceptor
	PrepareNextTurn(context.Context, *TurnContext) error
}

type ShouldStopInterceptor interface {
	Interceptor
	ShouldStop(context.Context, TurnContext) bool
}

// InterceptorFuncs is the convenient functional implementation. Nil callbacks
// are no-ops, so applications can configure only the lifecycle points they
// need without defining a new type.
type InterceptorFuncs struct {
	Name        string
	BeforeModel func(context.Context, *ModelRequest) error
	AfterModel  func(context.Context, *Message) error
	BeforeTool  func(context.Context, ToolCallContext) (ToolCallDecision, error)
	AfterTool   func(context.Context, ToolCallContext, *ToolResult) error
	PrepareTurn func(context.Context, *TurnContext) error
	Stop        func(context.Context, TurnContext) bool
}

func (i InterceptorFuncs) InterceptorName() string { return i.Name }

func (i InterceptorFuncs) BeforeModelCall(ctx context.Context, request *ModelRequest) error {
	if i.BeforeModel == nil {
		return nil
	}
	return i.BeforeModel(ctx, request)
}

func (i InterceptorFuncs) AfterModelCall(ctx context.Context, message *Message) error {
	if i.AfterModel == nil {
		return nil
	}
	return i.AfterModel(ctx, message)
}

func (i InterceptorFuncs) BeforeToolCall(ctx context.Context, call ToolCallContext) (ToolCallDecision, error) {
	if i.BeforeTool == nil {
		return ToolCallDecision{}, nil
	}
	return i.BeforeTool(ctx, call)
}

func (i InterceptorFuncs) AfterToolCall(ctx context.Context, call ToolCallContext, result *ToolResult) error {
	if i.AfterTool == nil {
		return nil
	}
	return i.AfterTool(ctx, call, result)
}

func (i InterceptorFuncs) PrepareNextTurn(ctx context.Context, turn *TurnContext) error {
	if i.PrepareTurn == nil {
		return nil
	}
	return i.PrepareTurn(ctx, turn)
}

func (i InterceptorFuncs) ShouldStop(ctx context.Context, turn TurnContext) bool {
	return i.Stop != nil && i.Stop(ctx, turn)
}

func (a *Agent) beforeModelCall(ctx context.Context, request *ModelRequest) error {
	for _, interceptor := range a.config.Interceptors {
		if target, ok := interceptor.(BeforeModelCallInterceptor); ok {
			if err := target.BeforeModelCall(ctx, request); err != nil {
				return interceptorError(interceptor, "before model call", err)
			}
		}
	}
	if a.config.Hooks.BeforeModelCall != nil {
		return a.config.Hooks.BeforeModelCall(ctx, request)
	}
	return nil
}

func (a *Agent) afterModelCall(ctx context.Context, message *Message) error {
	if a.config.Hooks.AfterModelCall != nil {
		if err := a.config.Hooks.AfterModelCall(ctx, message); err != nil {
			return err
		}
	}
	for index := len(a.config.Interceptors) - 1; index >= 0; index-- {
		interceptor := a.config.Interceptors[index]
		if target, ok := interceptor.(AfterModelCallInterceptor); ok {
			if err := target.AfterModelCall(ctx, message); err != nil {
				return interceptorError(interceptor, "after model call", err)
			}
		}
	}
	return nil
}

func (a *Agent) prepareNextTurn(ctx context.Context, turn *TurnContext) error {
	if a.config.Hooks.PrepareNextTurn != nil {
		if err := a.config.Hooks.PrepareNextTurn(ctx, turn); err != nil {
			return err
		}
	}
	for index := len(a.config.Interceptors) - 1; index >= 0; index-- {
		interceptor := a.config.Interceptors[index]
		if target, ok := interceptor.(PrepareNextTurnInterceptor); ok {
			if err := target.PrepareNextTurn(ctx, turn); err != nil {
				return interceptorError(interceptor, "prepare next turn", err)
			}
		}
	}
	return nil
}

func (a *Agent) shouldStop(ctx context.Context, turn TurnContext) bool {
	if a.config.Hooks.ShouldStop != nil && a.config.Hooks.ShouldStop(ctx, turn) {
		return true
	}
	for index := len(a.config.Interceptors) - 1; index >= 0; index-- {
		if target, ok := a.config.Interceptors[index].(ShouldStopInterceptor); ok && target.ShouldStop(ctx, turn) {
			return true
		}
	}
	return false
}

func (a *Agent) beforeToolCall(ctx context.Context, callContext ToolCallContext) (ToolCallContext, *ToolResult) {
	apply := func(decision ToolCallDecision, err error, owner string) *ToolResult {
		if err != nil {
			result := errorToolResult(owner + ": " + err.Error())
			return &result
		}
		if decision.Block {
			reason := decision.Reason
			if reason == "" {
				reason = "blocked by interceptor"
			}
			result := errorToolResult(reason)
			return &result
		}
		if len(decision.Arguments) > 0 {
			callContext.Call.Arguments = append([]byte(nil), decision.Arguments...)
		}
		return nil
	}
	for _, interceptor := range a.config.Interceptors {
		if target, ok := interceptor.(BeforeToolCallInterceptor); ok {
			decision, err := target.BeforeToolCall(ctx, callContext)
			if immediate := apply(decision, err, "before tool call interceptor "+interceptor.InterceptorName()); immediate != nil {
				return callContext, immediate
			}
		}
	}
	if a.config.Hooks.BeforeToolCall != nil {
		decision, err := a.config.Hooks.BeforeToolCall(ctx, callContext)
		if immediate := apply(decision, err, "before tool call hook"); immediate != nil {
			return callContext, immediate
		}
	}
	return callContext, nil
}

func (a *Agent) afterToolCall(ctx context.Context, callContext ToolCallContext, result *ToolResult, executed bool, attempts int) {
	callContext.Executed = executed
	callContext.Attempts = attempts
	if executed && a.config.Hooks.AfterToolCall != nil {
		if err := a.config.Hooks.AfterToolCall(ctx, callContext, result); err != nil {
			*result = errorToolResult("after tool call hook: " + err.Error())
		}
	}
	for index := len(a.config.Interceptors) - 1; index >= 0; index-- {
		interceptor := a.config.Interceptors[index]
		if target, ok := interceptor.(AfterToolCallInterceptor); ok {
			if err := target.AfterToolCall(ctx, callContext, result); err != nil {
				*result = errorToolResult(interceptorError(interceptor, "after tool call", err).Error())
			}
		}
	}
}

func interceptorError(interceptor Interceptor, phase string, err error) error {
	name := interceptor.InterceptorName()
	if name == "" {
		name = "unnamed"
	}
	return &LifecycleError{Interceptor: name, Phase: phase, Err: err}
}

// LifecycleError identifies which interceptor rejected a lifecycle stage.
type LifecycleError struct {
	Interceptor string
	Phase       string
	Err         error
}

func (e *LifecycleError) Error() string {
	return "agentcore: interceptor " + e.Interceptor + " failed during " + e.Phase + ": " + e.Err.Error()
}

func (e *LifecycleError) Unwrap() error { return e.Err }
