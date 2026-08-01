package agentcore

import (
	"context"
	"encoding/json"
	"errors"
)

// FuncTool turns functions into a Tool.
type FuncTool struct {
	ToolDefinition ToolDefinition
	Mode           ToolExecutionMode
	Policy         ToolPolicy
	ValidateFunc   func(json.RawMessage) error
	ExecuteFunc    func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error)
}

func (t FuncTool) Definition() ToolDefinition {
	return t.ToolDefinition
}

func (t FuncTool) ExecutionMode() ToolExecutionMode {
	return t.Mode
}

func (t FuncTool) ToolPolicy() ToolPolicy {
	return t.Policy
}

func (t FuncTool) Validate(arguments json.RawMessage) error {
	if t.ValidateFunc == nil {
		return nil
	}
	return t.ValidateFunc(arguments)
}

func (t FuncTool) Execute(ctx context.Context, arguments json.RawMessage, update ToolUpdateSink) (ToolResult, error) {
	if t.ExecuteFunc == nil {
		return ToolResult{}, errors.New("agentcore: FuncTool ExecuteFunc is required")
	}
	return t.ExecuteFunc(ctx, arguments, update)
}
