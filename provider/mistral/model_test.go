package mistral

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/z3r2ne/agentcore"
)

func TestMistralWireRequestAndNativeThinkingStream(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/chat/completions" {
			t.Errorf("path=%s", request.URL.Path)
		}
		if err := json.NewDecoder(request.Body).Decode(&captured); err != nil {
			t.Error(err)
		}
		response.Header().Set("content-type", "text/event-stream")
		_, _ = fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"thinking\",\"thinking\":[{\"text\":\"plan\"}]},{\"type\":\"text\",\"text\":\"done\"}]},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()

	model, err := New(Config{Model: "mistral-test", APIKey: "secret", BaseURL: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := model.Stream(context.Background(), agentcore.ModelRequest{
		SystemPrompt: "system",
		Messages: []agentcore.Message{
			{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{
				{Type: agentcore.ContentThinking, Text: "reason"},
				{Type: agentcore.ContentText, Text: "answer"},
				{Type: agentcore.ContentToolCall, ToolCall: &agentcore.ToolCall{ID: "call-long-id", Name: "lookup", Arguments: json.RawMessage(`{"q":"pi"}`)}},
			}},
			{Role: agentcore.RoleTool, ToolCallID: "call-long-id", ToolName: "lookup", Content: []agentcore.ContentBlock{
				{Type: agentcore.ContentText, Text: "found"},
				{Type: agentcore.ContentImage, MIMEType: "image/png", Data: []byte("image")},
			}},
		},
		Tools:   []agentcore.ToolDefinition{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Options: map[string]any{"maxTokens": 123, "promptMode": "reasoning"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var text, thinking string
	for {
		chunk, recvErr := stream.Recv()
		if recvErr != nil {
			if !errors.Is(recvErr, io.EOF) {
				t.Fatal(recvErr)
			}
			break
		}
		text += chunk.TextDelta
		thinking += chunk.ThinkingDelta
	}
	if text != "done" || thinking != "plan" {
		t.Fatalf("text=%q thinking=%q", text, thinking)
	}
	if captured["max_tokens"] != float64(123) || captured["prompt_mode"] != "reasoning" || captured["stream_options"] != nil {
		t.Fatalf("options=%#v", captured)
	}
	messages := captured["messages"].([]any)
	assistant := messages[1].(map[string]any)
	call := assistant["tool_calls"].([]any)[0].(map[string]any)
	if len(call["id"].(string)) != 9 || assistant["reasoning_content"] != nil || assistant["prefix"] != false {
		t.Fatalf("assistant=%#v", assistant)
	}
	tool := messages[2].(map[string]any)
	if tool["tool_call_id"] != call["id"] || len(tool["content"].([]any)) != 2 {
		t.Fatalf("tool=%#v call=%#v", tool, call)
	}
}

func TestNormalizeIDIsStableAndPreservesValidIDs(t *testing.T) {
	if got := normalizeID("abc123XYZ"); got != "abc123XYZ" {
		t.Fatalf("valid ID changed to %q", got)
	}
	first := normalizeID("provider-call-id")
	if len(first) != 9 || first != normalizeID("provider-call-id") || !validID.MatchString(first) {
		t.Fatalf("normalized ID=%q", first)
	}
}
