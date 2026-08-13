package pimessages

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/z3r2ne/agentcore"
)

func TestPiMessagesStream(t *testing.T) {
	payload := "data: {\"type\":\"text_start\",\"contentIndex\":0}\n\ndata: {\"type\":\"text_delta\",\"contentIndex\":0,\"delta\":\"hi\"}\n\ndata: {\"type\":\"text_end\",\"contentIndex\":0,\"content\":\"hi\",\"contentSignature\":\"sig\"}\n\ndata: {\"type\":\"done\",\"reason\":\"stop\",\"usage\":{\"input\":3,\"output\":1,\"cacheRead\":2}}\n\n"
	stream := newStream(context.Background(), io.NopCloser(strings.NewReader(payload)), 1024)
	first, err := stream.Recv()
	if err != nil || first.TextDelta != "hi" {
		t.Fatalf("first=%+v err=%v", first, err)
	}
	done, err := stream.Recv()
	if err != nil || done.StopReason != agentcore.StopReasonStop || done.Usage.CacheReadTokens != 2 || done.ProviderData == nil {
		t.Fatalf("done=%+v err=%v", done, err)
	}
	replayed, err := piMessage(agentcore.Message{Role: agentcore.RoleAssistant, StopReason: agentcore.StopReasonStop, ProviderData: done.ProviderData}, "radius-model")
	if err != nil {
		t.Fatal(err)
	}
	content := replayed["content"].([]any)
	if content[0].(map[string]any)["textSignature"] != "sig" {
		t.Fatalf("replayed=%#v", replayed)
	}
	_, err = stream.Recv()
	if !errors.Is(err, io.EOF) {
		t.Fatalf("err=%v", err)
	}
}

func TestPiMessagesToolResultPreservesImage(t *testing.T) {
	message, err := piMessage(agentcore.Message{Role: agentcore.RoleTool, ToolCallID: "c", ToolName: "look", Content: []agentcore.ContentBlock{
		{Type: agentcore.ContentText, Text: "found"},
		{Type: agentcore.ContentImage, MIMEType: "image/png", Data: []byte("image")},
	}}, "radius-model")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(message)
	if err != nil || !strings.Contains(string(encoded), `"type":"image"`) || !strings.Contains(string(encoded), `"data":"aW1hZ2U="`) {
		t.Fatalf("message=%s err=%v", encoded, err)
	}
}
