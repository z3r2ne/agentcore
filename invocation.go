package agentcore

import "context"

// ToolInvocation identifies one concrete tool attempt. It is attached to the
// context passed to Tool.Execute so adapters can implement stable idempotency,
// audit correlation, and execution-scoped policy without changing Tool's
// backwards-compatible method signature.
type ToolInvocation struct {
	Turn        int      `json:"turn"`
	Attempt     int      `json:"attempt"`
	MaxAttempts int      `json:"maxAttempts"`
	Call        ToolCall `json:"call"`
}

type toolInvocationContextKey struct{}

// ToolInvocationFromContext returns invocation metadata while Tool.Execute is
// running. Calls outside agentcore execution return ok=false.
func ToolInvocationFromContext(ctx context.Context) (invocation ToolInvocation, ok bool) {
	if ctx == nil {
		return ToolInvocation{}, false
	}
	invocation, ok = ctx.Value(toolInvocationContextKey{}).(ToolInvocation)
	if ok {
		invocation.Call = cloneToolCall(invocation.Call)
	}
	return invocation, ok
}

func withToolInvocation(ctx context.Context, invocation ToolInvocation) context.Context {
	invocation.Call = cloneToolCall(invocation.Call)
	return context.WithValue(ctx, toolInvocationContextKey{}, invocation)
}
