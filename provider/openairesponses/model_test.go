package openairesponses

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/z3r2ne/agentcore"
)

func TestResponsesStreamToolAndUsage(t *testing.T) {
	payload := strings.Join([]string{
		"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"work\",\"arguments\":\"\"}}\n\n",
		"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"c1\",\"delta\":\"{\\\"x\\\":1}\"}\n\n",
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"work\",\"arguments\":\"{\\\"x\\\":1}\"}],\"usage\":{\"input_tokens\":12,\"input_tokens_details\":{\"cached_tokens\":2},\"output_tokens\":3}}}\n\n",
	}, "")
	stream := newResponseStream(context.Background(), io.NopCloser(strings.NewReader(payload)), 4096)
	var args string
	var terminal agentcore.ModelChunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, call := range chunk.ToolCallDeltas {
			args += call.ArgumentsDelta
		}
		if chunk.StopReason != "" {
			terminal = chunk
		}
	}
	if args != `{"x":1}` || terminal.StopReason != agentcore.StopReasonToolUse || terminal.Usage.InputTokens != 10 || terminal.ProviderData == nil {
		t.Fatalf("args=%q terminal=%+v", args, terminal)
	}
}
func TestResponsesMessageConversion(t *testing.T) {
	items, err := responseItems(agentcore.Message{Role: agentcore.RoleAssistant, Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "ok"}, {Type: agentcore.ContentToolCall, ToolCall: &agentcore.ToolCall{ID: "c", Name: "work", Arguments: []byte(`{}`)}}}})
	if err != nil || len(items) != 2 {
		t.Fatalf("items=%#v err=%v", items, err)
	}
}

func TestResponsesIncompleteMaxOutputIsLength(t *testing.T) {
	payload := "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"partial\"}]}],\"usage\":{\"input_tokens\":4,\"output_tokens\":2}}}\n\n"
	stream := newResponseStream(context.Background(), io.NopCloser(strings.NewReader(payload)), 4096)
	chunk, err := stream.Recv()
	if err != nil || chunk.StopReason != agentcore.StopReasonLength || chunk.ProviderData == nil || chunk.Usage == nil {
		t.Fatalf("chunk=%+v err=%v", chunk, err)
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("terminal err=%v", err)
	}
}

func TestResponsesIncompleteContentFilterIsError(t *testing.T) {
	payload := "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"}}}\n\n"
	stream := newResponseStream(context.Background(), io.NopCloser(strings.NewReader(payload)), 4096)
	if _, err := stream.Recv(); err == nil || !strings.Contains(err.Error(), "content_filter") {
		t.Fatalf("err=%v", err)
	}
}
