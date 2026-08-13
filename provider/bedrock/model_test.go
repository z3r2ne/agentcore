package bedrock

import (
	"encoding/json"
	"testing"

	"github.com/z3r2ne/agentcore"
)

func TestBedrockInputConvertsToolsAndCoalescesResults(t *testing.T) {
	model := &Model{model: "test"}
	input, err := model.input(agentcore.ModelRequest{SystemPrompt: "system", Tools: []agentcore.ToolDefinition{{Name: "work", Parameters: json.RawMessage(`{"type":"object"}`)}}, Messages: []agentcore.Message{{Role: agentcore.RoleTool, ToolCallID: "a", ToolName: "work", Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "1"}}}, {Role: agentcore.RoleTool, ToolCallID: "b", ToolName: "work", Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "2"}}}}, Options: map[string]any{"maxTokens": 100}})
	if err != nil {
		t.Fatal(err)
	}
	if len(input.Messages) != 1 || len(input.Messages[0].Content) != 2 || input.ToolConfig == nil || input.InferenceConfig == nil || input.InferenceConfig.MaxTokens == nil || *input.InferenceConfig.MaxTokens != 100 {
		t.Fatalf("input=%+v", input)
	}
}
