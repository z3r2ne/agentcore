package google

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/z3r2ne/agentcore"
)

func TestGoogleRequestCoalescesFunctionResponses(t *testing.T) {
	model, _ := New(Config{Model: "gemini-test"})
	body, err := model.requestBody(agentcore.ModelRequest{Messages: []agentcore.Message{{Role: agentcore.RoleTool, ToolCallID: "a", ToolName: "one", Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "1"}}}, {Role: agentcore.RoleTool, ToolCallID: "b", ToolName: "two", Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "2"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	contents := body["contents"].([]map[string]any)
	if len(contents) != 1 || len(contents[0]["parts"].([]any)) != 2 {
		t.Fatalf("contents=%#v", contents)
	}
}
func TestGoogleStreamParsesNativeEvents(t *testing.T) {
	payload := "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"think\",\"thought\":true},{\"functionCall\":{\"id\":\"c1\",\"name\":\"work\",\"args\":{\"x\":1}}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"cachedContentTokenCount\":3,\"candidatesTokenCount\":2}}\n\n"
	stream := newGoogleStream(context.Background(), io.NopCloser(strings.NewReader(payload)), 4096)
	chunk, err := stream.Recv()
	if err != nil {
		t.Fatal(err)
	}
	if chunk.ThinkingDelta != "think" || len(chunk.ToolCallDeltas) != 1 || chunk.StopReason != agentcore.StopReasonToolUse || chunk.Usage.InputTokens != 7 || chunk.ProviderData == nil {
		t.Fatalf("chunk=%+v", chunk)
	}
	_, err = stream.Recv()
	if !errors.Is(err, io.EOF) {
		t.Fatalf("err=%v", err)
	}
}
