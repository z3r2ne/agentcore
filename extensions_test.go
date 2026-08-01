package agentcore

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestBuilderComposesSkillsToolsAndOrderedInterceptors(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "double", ArgumentsDelta: `{"value":1}`}}, StopReason: StopReasonToolUse}},
		{{TextDelta: "done", StopReason: StopReasonStop}},
	}}
	var order []string
	first := InterceptorFuncs{
		Name: "application-audit",
		BeforeTool: func(_ context.Context, call ToolCallContext) (ToolCallDecision, error) {
			order = append(order, "before:a")
			return ToolCallDecision{Arguments: json.RawMessage(`{"value":2}`)}, nil
		},
		AfterTool: func(_ context.Context, call ToolCallContext, result *ToolResult) error {
			order = append(order, "after:a")
			if !call.Executed || call.Attempts != 1 {
				t.Fatalf("call context=%+v", call)
			}
			result.Content = []ContentBlock{{Type: ContentText, Text: "audited:" + result.Text()}}
			return nil
		},
	}
	second := InterceptorFuncs{
		Name: "skill-policy",
		BeforeTool: func(_ context.Context, call ToolCallContext) (ToolCallDecision, error) {
			order = append(order, "before:b")
			if string(call.Call.Arguments) != `{"value":2}` {
				t.Fatalf("rewritten arguments were not passed down: %s", call.Call.Arguments)
			}
			return ToolCallDecision{}, nil
		},
		AfterTool: func(_ context.Context, _ ToolCallContext, _ *ToolResult) error {
			order = append(order, "after:b")
			return nil
		},
	}
	double := FuncTool{
		ToolDefinition: ToolDefinition{Name: "double", Parameters: json.RawMessage(`{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}`)},
		ExecuteFunc: func(_ context.Context, arguments json.RawMessage, _ ToolUpdateSink) (ToolResult, error) {
			if string(arguments) != `{"value":2}` {
				t.Fatalf("arguments=%s", arguments)
			}
			return TextToolResult("4"), nil
		},
	}
	skill := FuncSkill{
		SkillDefinition: SkillDefinition{Name: "math", Version: "1", Description: "basic math"},
		Content:         SkillContent{Instructions: "Always verify arithmetic.", Tools: []Tool{double}, Interceptors: []Interceptor{second}},
	}
	agent, err := NewBuilder(model).
		SystemPrompt("You are useful.").
		Use(first).
		Skills(skill).
		Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "double")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []string{"before:a", "before:b", "after:b", "after:a"}) {
		t.Fatalf("order=%v", order)
	}
	if got := result.State.Messages[2].Text(); got != "audited:4" {
		t.Fatalf("tool result=%q", got)
	}
	if len(model.requests) != 2 || !strings.Contains(model.requests[0].SystemPrompt, `<skill name="math" version="1">`) || !strings.Contains(model.requests[0].SystemPrompt, "Always verify arithmetic.") {
		t.Fatalf("system prompt=%q", model.requests[0].SystemPrompt)
	}
}

func TestAfterToolInterceptorObservesBlockedCalls(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{
		{{ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "danger", ArgumentsDelta: `{}`}}, StopReason: StopReasonToolUse}},
		{{TextDelta: "blocked", StopReason: StopReasonStop}},
	}}
	executed := false
	afterCalled := false
	tool := FuncTool{ToolDefinition: ToolDefinition{Name: "danger"}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
		executed = true
		return TextToolResult("unsafe"), nil
	}}
	guard := InterceptorFuncs{
		Name: "approval",
		BeforeTool: func(context.Context, ToolCallContext) (ToolCallDecision, error) {
			return ToolCallDecision{Block: true, Reason: "approval required"}, nil
		},
		AfterTool: func(_ context.Context, call ToolCallContext, result *ToolResult) error {
			afterCalled = true
			if call.Executed || !result.IsError || result.Text() != "approval required" {
				t.Fatalf("call=%+v result=%+v", call, result)
			}
			return nil
		},
	}
	agent, err := NewBuilder(model).Tools(tool).Use(guard).Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "run")}, nil); err != nil {
		t.Fatal(err)
	}
	if executed || !afterCalled {
		t.Fatalf("executed=%t afterCalled=%t", executed, afterCalled)
	}
}

func TestBuilderRejectsDuplicateSkillsBeforeAgentStarts(t *testing.T) {
	model := &fakeModel{}
	one := FuncSkill{SkillDefinition: SkillDefinition{Name: "review"}}
	two := FuncSkill{SkillDefinition: SkillDefinition{Name: "review"}}
	if _, err := NewBuilder(model).Skills(one, two).Build(context.Background()); err == nil || !strings.Contains(err.Error(), "duplicate skill") {
		t.Fatalf("err=%v", err)
	}
}
