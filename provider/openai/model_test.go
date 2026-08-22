package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/z3r2ne/agentcore"
)

func TestModelRunsParallelToolLoopAndPreservesProviderData(t *testing.T) {
	var requests atomic.Int32
	var active atomic.Int32
	var peak atomic.Int32
	handlerErrors := make(chan error, 4)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		call := requests.Add(1)
		if request.URL.Path != "/custom/v1/chat/completions" || request.URL.Query().Get("api-version") != "test" {
			handlerErrors <- fmt.Errorf("unexpected URL %s", request.URL.String())
			response.WriteHeader(http.StatusBadRequest)
			return
		}
		if got := request.Header.Get("Authorization"); got != "Custom token" {
			handlerErrors <- fmt.Errorf("authorization = %q", got)
		}
		if request.Header.Get("X-String") != "one" || len(request.Header.Values("X-Multi")) != 2 {
			handlerErrors <- fmt.Errorf("headers = %#v", request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			handlerErrors <- err
			response.WriteHeader(http.StatusBadRequest)
			return
		}
		if body["model"] != "gpt-test" || body["stream"] != true || body["n"] != nil || body["temperature"] != float64(0) {
			handlerErrors <- fmt.Errorf("request body = %#v", body)
		}
		messages, _ := body["messages"].([]any)
		if call == 1 {
			if len(messages) != 2 || messages[0].(map[string]any)["role"] != "system" || len(body["tools"].([]any)) != 1 || body["parallel_tool_calls"] != true {
				handlerErrors <- fmt.Errorf("first request body = %#v", body)
			}
		} else {
			if len(messages) != 5 {
				handlerErrors <- fmt.Errorf("second request messages = %#v", messages)
			} else {
				assistant := messages[2].(map[string]any)
				if assistant["reasoning_content"] != "plan both" || assistant["vendor_field"].(map[string]any)["signature"] != "sig" || len(assistant["tool_calls"].([]any)) != 2 {
					handlerErrors <- fmt.Errorf("preserved assistant = %#v", assistant)
				}
				if messages[3].(map[string]any)["tool_call_id"] != "call-1" || messages[4].(map[string]any)["tool_call_id"] != "call-2" {
					handlerErrors <- fmt.Errorf("tool result order = %#v", messages[3:])
				}
			}
		}
		response.Header().Set("Content-Type", "text/event-stream")
		flusher := response.(http.Flusher)
		writeSSE := func(payload string) {
			_, _ = fmt.Fprintf(response, "data: %s\n\n", payload)
			flusher.Flush()
		}
		if call == 1 {
			// The first event uses multiple data fields to exercise SSE framing.
			_, _ = io.WriteString(response, "data: {\"id\":\"chat-1\",\"model\":\"gpt-test\",\n")
			_, _ = io.WriteString(response, "data: \"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"plan \",\"vendor_field\":{\"signature\":\"sig\"},\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"double\",\"arguments\":\"{\\\"value\\\":\"}},{\"index\":1,\"id\":\"call-2\",\"type\":\"function\",\"function\":{\"name\":\"double\",\"arguments\":\"{\\\"value\\\":\"}}]},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			writeSSE(`{"choices":[{"index":0,"delta":{"reasoning_content":"both","tool_calls":[{"index":0,"function":{"arguments":"2}"}},{"index":1,"function":{"arguments":"3}"}}]},"finish_reason":"tool_calls"}]}`)
			writeSSE(`{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2},"cache_creation_input_tokens":1}}`)
		} else {
			writeSSE(`{"id":"chat-2","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}`)
			writeSSE(`{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":1}}`)
		}
		writeSSE(`[DONE]`)
	}))
	defer server.Close()

	stringHeaders := map[string]string{"X-String": "one"}
	multiHeaders := http.Header{"X-Multi": {"first", "second"}, "Authorization": {"Custom token"}}
	model, err := New(Config{
		Model: "gpt-test", BaseURL: server.URL + "/custom/v1?api-version=test", APIKey: "ignored",
		Headers: stringHeaders, Header: multiHeaders,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rendered := fmt.Sprintf("%+v %#v", model, model); strings.Contains(rendered, "ignored") || strings.Contains(rendered, "Custom token") {
		t.Fatalf("model formatting exposed credentials: %s", rendered)
	}
	// New must have copied caller-owned configuration.
	stringHeaders["X-String"] = "changed"
	multiHeaders.Set("Authorization", "changed")

	tool := agentcore.FuncTool{
		ToolDefinition: agentcore.ToolDefinition{
			Name: "double", Parameters: json.RawMessage(`{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}`),
		},
		ExecuteFunc: func(_ context.Context, arguments json.RawMessage, _ agentcore.ToolUpdateSink) (agentcore.ToolResult, error) {
			current := active.Add(1)
			for {
				observed := peak.Load()
				if current <= observed || peak.CompareAndSwap(observed, current) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			active.Add(-1)
			var input struct {
				Value int `json:"value"`
			}
			if err := json.Unmarshal(arguments, &input); err != nil {
				return agentcore.ToolResult{}, err
			}
			return agentcore.TextToolResult(fmt.Sprint(input.Value * 2)), nil
		},
	}
	agent, err := agentcore.New(agentcore.Config{
		Model: model, SystemPrompt: "system", Tools: []agentcore.Tool{tool},
		ModelOptions: map[string]any{"temperature": 0, "parallel_tool_calls": true, "model": "must-not-override", "n": 5},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), agentcore.State{}, []agentcore.Message{
		agentcore.TextMessage(agentcore.RoleUser, "double two and three"),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	close(handlerErrors)
	for handlerErr := range handlerErrors {
		t.Error(handlerErr)
	}
	if requests.Load() != 2 || peak.Load() != 2 {
		t.Fatalf("requests=%d peak tool concurrency=%d", requests.Load(), peak.Load())
	}
	if result.StopReason != agentcore.StopReasonStop || result.State.Messages[len(result.State.Messages)-1].Text() != "done" {
		t.Fatalf("result = %+v", result)
	}
	if result.Usage != (agentcore.Usage{InputTokens: 19, OutputTokens: 5, CacheReadTokens: 2, CacheWriteTokens: 1}) {
		t.Fatalf("usage = %+v", result.Usage)
	}
	firstAssistant := result.State.Messages[1]
	if firstAssistant.ProviderData == nil || firstAssistant.ProviderData.Format != ProviderDataFormat {
		t.Fatalf("provider data = %#v", firstAssistant.ProviderData)
	}
	if firstAssistant.Text() != "" || messageContent(firstAssistant.Content, agentcore.ContentThinking) != "plan both" || len(firstAssistant.ToolCalls()) != 2 {
		t.Fatalf("first assistant = %#v", firstAssistant)
	}
	serialized, err := json.Marshal(firstAssistant)
	if err != nil {
		t.Fatal(err)
	}
	var restored agentcore.Message
	if err := json.Unmarshal(serialized, &restored); err != nil {
		t.Fatal(err)
	}
	converted, err := convertMessage(restored)
	if err != nil || converted["reasoning_content"] != "plan both" || converted["vendor_field"].(map[string]any)["signature"] != "sig" {
		t.Fatalf("restored provider message = %#v, err=%v", converted, err)
	}
}

func TestConvertMessageFallsBackFromMalformedPreservedToolArguments(t *testing.T) {
	message := agentcore.Message{
		Role: agentcore.RoleAssistant,
		Content: []agentcore.ContentBlock{{Type: agentcore.ContentToolCall, ToolCall: &agentcore.ToolCall{
			ID: "call-1", Name: "phone_action", Arguments: json.RawMessage(`{"action":"tap"}`),
		}}},
		ProviderData: &agentcore.ProviderData{
			Format: ProviderDataFormat,
			Data: json.RawMessage(`{
				"source":"test-scope",
				"message":{"role":"assistant","content":null,"tool_calls":[{
					"id":"call-1","type":"function","function":{"name":"phone_action","arguments":"{\"action\":\"tap\""}
				}]}
			}`),
		},
	}

	converted, err := convertMessageForScope(message, "test-scope")
	if err != nil {
		t.Fatal(err)
	}
	calls, ok := converted["tool_calls"].([]map[string]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %#v", converted["tool_calls"])
	}
	function := calls[0]["function"].(map[string]any)
	if function["arguments"] != `{"action":"tap"}` {
		t.Fatalf("function = %#v", function)
	}
	if converted["content"] != "" {
		t.Fatalf("content = %#v", converted["content"])
	}
}

func TestPreservedAssistantToolArgumentsValidation(t *testing.T) {
	tests := []struct {
		name    string
		message string
		valid   bool
	}{
		{name: "no tool call", message: `{"content":"done"}`, valid: true},
		{name: "valid tool call", message: `{"tool_calls":[{"function":{"arguments":"{}"}}]}`, valid: true},
		{name: "invalid tool call JSON", message: `{"tool_calls":[{"function":{"arguments":"{\"path\":"}}]}`, valid: false},
		{name: "non-string arguments", message: `{"tool_calls":[{"function":{"arguments":{}}}]}`, valid: false},
		{name: "invalid legacy function call", message: `{"function_call":{"arguments":"["}}`, valid: false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var message map[string]any
			if err := json.Unmarshal([]byte(test.message), &message); err != nil {
				t.Fatal(err)
			}
			if got := preservedAssistantToolArgumentsValid(message); got != test.valid {
				t.Fatalf("valid = %t, want %t", got, test.valid)
			}
		})
	}
}

func TestModelRetryUsesStructuredProviderError(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			response.Header().Set("Retry-After", "2")
			response.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(response, `{"error":{"message":"busy","type":"server_error","code":"overloaded"}}`)
			return
		}
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	model, err := New(Config{Model: "test", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	agent, err := agentcore.New(agentcore.Config{Model: model, ModelRetry: agentcore.RetryPolicy{
		MaxAttempts: 2, InitialDelay: time.Millisecond, ShouldRetry: IsRetryable,
	}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), agentcore.State{}, []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "go")}, nil)
	if err != nil || calls.Load() != 2 || result.State.Messages[len(result.State.Messages)-1].Text() != "ok" {
		t.Fatalf("calls=%d result=%+v err=%v", calls.Load(), result, err)
	}
}

func TestHTTPErrorIsBoundedAndClassified(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Retry-After", "3")
		response.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(response, `{"error":{"message":"`+strings.Repeat("x", 100)+`","type":"rate_limit","code":"slow_down"}}`)
	}))
	defer server.Close()
	model, err := New(Config{Model: "test", BaseURL: server.URL, MaxErrorBodyBytes: 48})
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Stream(context.Background(), agentcore.ModelRequest{})
	var providerError *Error
	if !errors.As(err, &providerError) {
		t.Fatalf("error type = %T (%v)", err, err)
	}
	if providerError.StatusCode != http.StatusTooManyRequests || !providerError.Retryable || providerError.RetryAfter != 3*time.Second {
		t.Fatalf("provider error = %#v", providerError)
	}
	if len(providerError.Body) > 70 || !strings.Contains(providerError.Body, "truncated") || !IsRetryable(err) {
		t.Fatalf("unbounded or unclassified error: %#v", providerError)
	}
}

func TestBadRequestIsNotRetryable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(response, `{"error":{"message":"bad option"}}`)
	}))
	defer server.Close()
	model, _ := New(Config{Model: "test", BaseURL: server.URL})
	_, err := model.Stream(context.Background(), agentcore.ModelRequest{})
	if err == nil || IsRetryable(err) {
		t.Fatalf("err = %v retryable=%t", err, IsRetryable(err))
	}
}

func TestRequestCancellationBeforeHeaders(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		close(started)
		<-release
	}))
	defer server.Close()
	model, _ := New(Config{Model: "test", BaseURL: server.URL})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := model.Stream(ctx, agentcore.ModelRequest{})
		done <- err
	}()
	<-started
	cancel()
	err := <-done
	close(release)
	if !errors.Is(err, context.Canceled) || IsRetryable(err) {
		t.Fatalf("err = %v", err)
	}
}

func TestRequestConversionAndConfigurationValidation(t *testing.T) {
	invalid := []Config{
		{},
		{Model: "x", BaseURL: "ftp://example.com"},
		{Model: "x", MaxResponseBodyBytes: -1},
	}
	for _, config := range invalid {
		if _, err := New(config); err == nil {
			t.Fatalf("New(%+v) succeeded", config)
		}
	}
	model, err := NewModel(Config{BaseURL: "https://example.test/v1/chat/completions"}, "x")
	if err != nil || model.endpoint != "https://example.test/v1/chat/completions" {
		t.Fatalf("endpoint=%q err=%v", model.endpoint, err)
	}
	_, err = model.requestBody(agentcore.ModelRequest{Messages: []agentcore.Message{{Role: "invalid"}}})
	if err == nil {
		t.Fatal("unsupported role succeeded")
	}
	_, err = model.requestBody(agentcore.ModelRequest{Tools: []agentcore.ToolDefinition{{Name: "bad", Parameters: json.RawMessage(`{`)}}})
	if err == nil {
		t.Fatal("invalid tool schema succeeded")
	}
	body, err := model.requestBody(agentcore.ModelRequest{Messages: []agentcore.Message{{
		Role:    agentcore.RoleAssistant,
		Content: []agentcore.ContentBlock{{Type: agentcore.ContentThinking, Text: "think"}, {Type: agentcore.ContentText, Text: "answer"}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	assistant := body["messages"].([]map[string]any)[0]
	if assistant["reasoning_content"] != "think" || assistant["content"] != "answer" {
		t.Fatalf("assistant request = %#v", assistant)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (function roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func TestCustomHTTPClient(t *testing.T) {
	var called atomic.Bool
	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		called.Store(true)
		if request.URL.String() != "https://compatible.example/v1/chat/completions" {
			return nil, fmt.Errorf("URL = %s", request.URL)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")),
			Request:    request,
		}, nil
	})}
	model, err := New(Config{Model: "test", BaseURL: "https://compatible.example/v1", HTTPClient: client})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := model.Stream(context.Background(), agentcore.ModelRequest{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = stream.Recv()
	if err == nil {
		_, err = stream.Recv()
	}
	if !called.Load() || !errors.Is(err, io.EOF) {
		t.Fatalf("called=%t err=%v", called.Load(), err)
	}
}
