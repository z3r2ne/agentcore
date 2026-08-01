package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/z3r2ne/agentcore"
)

func TestCancellationWhileReceivingStream(t *testing.T) {
	firstSent := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"},\"finish_reason\":null}]}\n\n")
		response.(http.Flusher).Flush()
		close(firstSent)
		<-release
	}))
	defer server.Close()
	model, _ := New(Config{Model: "test", BaseURL: server.URL})
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := model.Stream(ctx, agentcore.ModelRequest{})
	if err != nil {
		t.Fatal(err)
	}
	chunk, err := stream.Recv()
	if err != nil || chunk.TextDelta != "first" {
		t.Fatalf("chunk=%+v err=%v", chunk, err)
	}
	<-firstSent
	cancel()
	_, err = stream.Recv()
	close(release)
	if !errors.Is(err, context.Canceled) || IsRetryable(err) {
		t.Fatalf("err = %v", err)
	}
	_ = stream.Close()
}

func TestCloseUnblocksReceive(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		response.WriteHeader(http.StatusOK)
		response.(http.Flusher).Flush()
		<-release
	}))
	defer server.Close()
	model, _ := New(Config{Model: "test", BaseURL: server.URL})
	stream, err := model.Stream(context.Background(), agentcore.ModelRequest{})
	if err != nil {
		t.Fatal(err)
	}
	received := make(chan error, 1)
	go func() {
		_, err := stream.Recv()
		received <- err
	}()
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	close(release)
	if err := <-received; err == nil {
		t.Fatal("Recv succeeded after Close")
	}
}

func TestResponseBodyLimit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: "+strings.Repeat("x", 200)+"\n\n")
	}))
	defer server.Close()
	model, _ := New(Config{Model: "test", BaseURL: server.URL, MaxResponseBodyBytes: 32, MaxSSEEventBytes: 1024})
	stream, err := model.Stream(context.Background(), agentcore.ModelRequest{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = stream.Recv()
	if err == nil || !strings.Contains(err.Error(), "configured limit") || !IsRetryable(err) {
		t.Fatalf("err = %v", err)
	}
}

func TestSSEEventLimit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: "+strings.Repeat("x", 100)+"\n\n")
	}))
	defer server.Close()
	model, _ := New(Config{Model: "test", BaseURL: server.URL, MaxResponseBodyBytes: 1024, MaxSSEEventBytes: 32})
	stream, err := model.Stream(context.Background(), agentcore.ModelRequest{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = stream.Recv()
	if err == nil || !strings.Contains(err.Error(), "SSE event exceeds 32 bytes") {
		t.Fatalf("err = %v", err)
	}
}

func TestMalformedAndProviderStreamErrors(t *testing.T) {
	for name, payload := range map[string]string{
		"malformed": `not-json`,
		"provider":  `{"error":{"message":"stream failed","type":"server_error","code":"broken"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			stream := newStream(context.Background(), io.NopCloser(strings.NewReader("data: "+payload+"\n\n")), 1024)
			_, err := stream.Recv()
			if err == nil || !IsRetryable(err) {
				t.Fatalf("err = %v", err)
			}
			if name == "provider" {
				var providerError *Error
				if !errors.As(err, &providerError) || providerError.Code != "broken" || providerError.Type != "server_error" {
					t.Fatalf("provider error = %#v", providerError)
				}
			}
		})
	}
}

func TestStopReasonMapping(t *testing.T) {
	tests := map[string]agentcore.StopReason{
		"stop": agentcore.StopReasonStop, "end_turn": agentcore.StopReasonStop,
		"tool_calls": agentcore.StopReasonToolUse, "function_call": agentcore.StopReasonToolUse,
		"length": agentcore.StopReasonLength, "max_tokens": agentcore.StopReasonLength,
		"content_filter": agentcore.StopReasonError, "cancelled": agentcore.StopReasonAborted,
		"vendor_reason": agentcore.StopReason("vendor_reason"),
	}
	for input, expected := range tests {
		if actual := stopReason(input); actual != expected {
			t.Errorf("stopReason(%q) = %q, want %q", input, actual, expected)
		}
	}
}

func TestStreamWithoutDoneEndsAtEOF(t *testing.T) {
	payload := fmt.Sprintf("data: %s\n\n", `{"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`)
	stream := newStream(context.Background(), io.NopCloser(strings.NewReader(payload)), 1024)
	chunk, err := stream.Recv()
	if err != nil || chunk.TextDelta != "ok" || chunk.StopReason != agentcore.StopReasonStop {
		t.Fatalf("chunk=%+v err=%v", chunk, err)
	}
	_, err = stream.Recv()
	if !errors.Is(err, io.EOF) {
		t.Fatalf("err = %v", err)
	}
}

func TestInlineImageConversion(t *testing.T) {
	content, err := convertContent([]agentcore.ContentBlock{
		{Type: agentcore.ContentText, Text: "inspect"},
		{Type: agentcore.ContentImage, Data: []byte{1, 2, 3}, MIMEType: "image/png"},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := content.([]map[string]any)
	url := parts[1]["image_url"].(map[string]any)["url"].(string)
	if url != "data:image/png;base64,AQID" {
		t.Fatalf("URL = %q", url)
	}
}
