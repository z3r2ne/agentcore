package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/z3r2ne/agentcore"
)

func TestRequestCoalescesParallelToolResults(t *testing.T) {
	model, _ := New(Config{Model: "claude-test"})
	body, err := model.requestBody(agentcore.ModelRequest{Messages: []agentcore.Message{
		{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{{Type: agentcore.ContentToolCall, ToolCall: &agentcore.ToolCall{ID: "a", Name: "one", Arguments: json.RawMessage(`{}`)}}, {Type: agentcore.ContentToolCall, ToolCall: &agentcore.ToolCall{ID: "b", Name: "two", Arguments: json.RawMessage(`{}`)}}}},
		{Role: agentcore.RoleTool, ToolCallID: "a", ToolName: "one", Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "1"}}},
		{Role: agentcore.RoleTool, ToolCallID: "b", ToolName: "two", Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "2"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]map[string]any)
	if len(messages) != 2 || messages[1]["role"] != "user" || len(messages[1]["content"].([]any)) != 2 {
		t.Fatalf("messages=%#v", messages)
	}
}

func TestStreamParsesThinkingToolsUsageAndTerminal(t *testing.T) {
	payload := strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":2}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call-1\",\"name\":\"work\",\"input\":{}}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"x\\\":1}\"}}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":4}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}, "")
	stream := newStream(context.Background(), io.NopCloser(strings.NewReader(payload)), 4096)
	var thinking, args string
	var terminal agentcore.ModelChunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		thinking += chunk.ThinkingDelta
		for _, call := range chunk.ToolCallDeltas {
			args += call.ArgumentsDelta
		}
		if chunk.StopReason != "" {
			terminal = chunk
		}
	}
	if thinking != "plan" || args != `{"x":1}` || terminal.StopReason != agentcore.StopReasonToolUse || terminal.Usage == nil || terminal.Usage.InputTokens != 8 || terminal.ProviderData == nil {
		t.Fatalf("thinking=%q args=%q terminal=%+v", thinking, args, terminal)
	}
}
