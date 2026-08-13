package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func TestToolResultSinkFailureStillReturnsValidHistory(t *testing.T) {
	model := &fakeModel{responses: [][]ModelChunk{{{
		ToolCallDeltas: []ToolCallDelta{{Index: 0, ID: "call-1", Name: "one", ArgumentsDelta: `{}`}, {Index: 1, ID: "call-2", Name: "two", ArgumentsDelta: `{}`}},
		StopReason:     StopReasonToolUse,
	}}}}
	tool := func(name string) Tool {
		return FuncTool{ToolDefinition: ToolDefinition{Name: name}, ExecuteFunc: func(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error) {
			return TextToolResult(name), nil
		}}
	}
	agent, err := New(Config{Model: model, Tools: []Tool{tool("one"), tool("two")}})
	if err != nil {
		t.Fatal(err)
	}
	sinkErr := errors.New("observer failed")
	result, runErr := agent.Prompt(context.Background(), State{}, []Message{TextMessage(RoleUser, "go")}, func(_ context.Context, event Event) error {
		if event.Type == EventMessageStart && event.Message != nil && event.Message.Role == RoleTool {
			return sinkErr
		}
		return nil
	})
	if !errors.Is(runErr, sinkErr) {
		t.Fatalf("err=%v", runErr)
	}
	if err := ValidateHistory(result.State.Messages); err != nil {
		t.Fatalf("invalid returned history: %v; %+v", err, result.State.Messages)
	}
	if len(result.State.Messages) != 4 {
		t.Fatalf("messages=%+v", result.State.Messages)
	}
}

func TestRepairHistoryPreservesValidHistory(t *testing.T) {
	messages := []Message{
		TextMessage(RoleUser, "go"),
		{ID: "assistant-1", Role: RoleAssistant, StopReason: StopReasonToolUse, Content: []ContentBlock{
			{Type: ContentToolCall, ToolCall: &ToolCall{ID: "call-1", Name: "one", Arguments: json.RawMessage(`{"x":1}`)}},
			{Type: ContentToolCall, ToolCall: &ToolCall{ID: "call-2", Name: "two", Arguments: json.RawMessage(`{}`)}},
		}},
		{ID: "result-1", Role: RoleTool, ToolCallID: "call-1", ToolName: "one", Content: []ContentBlock{{Type: ContentText, Text: "1"}}},
		{ID: "result-2", Role: RoleTool, ToolCallID: "call-2", ToolName: "two", Content: []ContentBlock{{Type: ContentText, Text: "2"}}},
	}
	repaired, report := RepairHistory(messages)
	if report.Changed || !reflect.DeepEqual(repaired, messages) {
		t.Fatalf("report=%+v repaired=%+v", report, repaired)
	}
	if err := ValidateHistory(repaired); err != nil {
		t.Fatal(err)
	}
}

func TestRepairHistoryDropsPartialAttemptsAndRepairsToolFrames(t *testing.T) {
	messages := []Message{
		TextMessage(RoleUser, "start"),
		{ID: "partial", Role: RoleAssistant, IsError: true, StopReason: StopReasonError, ProviderData: &ProviderData{Format: "vendor", Data: json.RawMessage(`{"partial":true}`)}, Content: []ContentBlock{{Type: ContentToolCall, ToolCall: &ToolCall{ID: "bad", Name: "write", Arguments: json.RawMessage(`{"half"`)}}}},
		{ID: "assistant-2", Role: RoleAssistant, StopReason: StopReasonToolUse, Content: []ContentBlock{
			{Type: ContentToolCall, ToolCall: &ToolCall{ID: "call-1", Name: "first", Arguments: json.RawMessage(`{}`)}},
			{Type: ContentToolCall, ToolCall: &ToolCall{ID: "call-2", Name: "second", Arguments: json.RawMessage(`{}`)}},
		}},
		{Role: RoleTool, ToolCallID: "call-2", ToolName: "second", Content: []ContentBlock{{Type: ContentText, Text: "two"}}},
		{Role: RoleTool, ToolCallID: "orphan", ToolName: "other", Content: []ContentBlock{{Type: ContentText, Text: "bad"}}},
		TextMessage(RoleUser, "continue"),
	}
	repaired, report := RepairHistory(messages)
	if !report.Changed || len(repaired) != 5 {
		t.Fatalf("report=%+v repaired=%+v", report, repaired)
	}
	if repaired[1].ID != "assistant-2" || repaired[2].ToolCallID != "call-1" || !repaired[2].IsError || repaired[3].ToolCallID != "call-2" || repaired[3].Text() != "two" {
		t.Fatalf("repaired frame=%+v", repaired[1:4])
	}
	if repaired[4].Role != RoleUser || repaired[4].Text() != "continue" {
		t.Fatalf("tail=%+v", repaired[4:])
	}
	if err := ValidateHistory(repaired); err != nil {
		t.Fatal(err)
	}
	again, secondReport := RepairHistory(repaired)
	if secondReport.Changed || !reflect.DeepEqual(again, repaired) {
		t.Fatalf("repair is not idempotent: report=%+v", secondReport)
	}
}

func TestRepairHistoryNormalizesMalformedIdentityDeterministically(t *testing.T) {
	messages := []Message{{ID: "assistant-x", Role: RoleAssistant, Content: []ContentBlock{
		{Type: ContentToolCall, ToolCall: &ToolCall{Name: "one", Arguments: json.RawMessage(`{}`)}},
		{Type: ContentToolCall, ToolCall: &ToolCall{Name: "two", Arguments: json.RawMessage(`{}`)}},
	}}}
	first, _ := RepairHistory(messages)
	second, _ := RepairHistory(messages)
	if !reflect.DeepEqual(first, second) || first[0].ToolCalls()[0].ID == first[0].ToolCalls()[1].ID {
		t.Fatalf("first=%+v second=%+v", first, second)
	}
	if err := ValidateHistory(first); err != nil {
		t.Fatal(err)
	}
}
