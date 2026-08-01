package agentcore

import "errors"

var (
	// ErrModelRequired indicates an Agent was configured without a model.
	ErrModelRequired = errors.New("agentcore: model is required")
	// ErrMaxTurns indicates the configured model-turn limit was reached.
	ErrMaxTurns = errors.New("agentcore: maximum turns exceeded")
	// ErrSessionBusy indicates a second run was started on an active Session.
	ErrSessionBusy = errors.New("agentcore: session is already running")
	// ErrSessionIdle indicates a running-only operation was requested while idle.
	ErrSessionIdle = errors.New("agentcore: session is idle")
)
