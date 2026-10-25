package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/z3r2ne/agentcore"
)

func TestStreamAccumulatesIndexedReasoningDetails(t *testing.T) {
	deltas := []string{
		`{"reasoning_details":[{"index":0,"type":"reasoning.text","text":"Plan ","signature":null,"id":"r0","format":"test-v1","vendor":{"keep":true}}],"vendor_array":["first"]}`,
		`{"reasoning_details":[{"index":1,"type":"reasoning.summary","summary":"Summary "},{"index":0,"text":"完成"}],"vendor_array":["second"]}`,
		`{"reasoning_details":[{"index":0,"signature":"sig"},{"index":1,"summary":"done"},{"index":2,"type":"reasoning.encrypted","data":"YWJ"}]}`,
		`{"reasoning_details":[{"index":0,"signature":"nature"},{"index":2,"data":"j"}],"content":"ok"}`,
	}
	var events strings.Builder
	for _, delta := range deltas {
		fmt.Fprintf(&events, "data: {\"choices\":[{\"index\":0,\"delta\":%s}]}\n\n", delta)
	}
	events.WriteString("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	stream := newStream(context.Background(), io.NopCloser(strings.NewReader(events.String())), 4096)
	defer stream.Close()
	var providerData *agentcore.ProviderData
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if chunk.ProviderData != nil {
			providerData = chunk.ProviderData
		}
	}
	if providerData == nil {
		t.Fatal("terminal chunk did not preserve provider data")
	}
	expected := decodeJSONValue(json.RawMessage(`[
		{"index":0,"type":"reasoning.text","text":"Plan 完成","signature":"signature","id":"r0","format":"test-v1","vendor":{"keep":true}},
		{"index":1,"type":"reasoning.summary","summary":"Summary done"},
		{"index":2,"type":"reasoning.encrypted","data":"YWJj"}
	]`))
	actual := decodeJSONValue(providerData.Data).(map[string]any)["message"].(map[string]any)
	if !reflect.DeepEqual(actual["reasoning_details"], expected) {
		t.Fatalf("reasoning_details = %#v, want %#v", actual["reasoning_details"], expected)
	}
	if actual["content"] != "ok" || !reflect.DeepEqual(actual["vendor_array"], []any{"first", "second"}) {
		t.Fatalf("other provider fields changed: %#v", actual)
	}
}

func TestReasoningDetailsKeepAmbiguousBlocksSeparate(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"unindexed", `[{"type":"reasoning.text","text":"a"},{"type":"reasoning.text","text":"b"}]`},
		{"null index", `[{"index":null,"text":"a"},{"index":null,"text":"b"}]`},
		{"string index", `[{"index":"0","text":"a"},{"index":"0","text":"b"}]`},
		{"negative index", `[{"index":-1,"text":"a"},{"index":-1,"text":"b"}]`},
		{"fractional index", `[{"index":0.5,"text":"a"},{"index":0.5,"text":"b"}]`},
		{"overflow index", `[{"index":9223372036854775808,"text":"a"},{"index":9223372036854775808,"text":"b"}]`},
		{"distinct indexes", `[{"index":0,"text":"a"},{"index":1,"text":"b"}]`},
		{"distinct IDs", `[{"index":0,"id":"a","text":"a"},{"index":0,"id":"b","text":"b"}]`},
		{"distinct formats", `[{"index":0,"format":"a","text":"a"},{"index":0,"format":"b","text":"b"}]`},
		{"distinct types", `[{"index":0,"type":"reasoning.text","text":"a"},{"index":0,"type":"reasoning.summary","summary":"b"}]`},
		{"unknown type", `[{"index":0,"type":"reasoning.vendor","data":"a"},{"index":0,"type":"reasoning.vendor","data":"b"}]`},
		{"opaque barrier", `[{"index":0,"text":"a"},null,{"index":0,"text":"b"}]`},
		{"unindexed barrier", `[{"index":0,"text":"a"},{"text":"b"},{"index":0,"text":"c"}]`},
		{"conflicting block barrier", `[{"index":0,"id":"a","text":"a"},{"index":0,"id":"b","text":"b"},{"index":0,"id":"a","text":"c"}]`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := decodeJSONValue(json.RawMessage(test.input)).([]any)
			before := cloneJSONSlice(input)
			actual := mergeReasoningDetails(nil, input)
			if !reflect.DeepEqual(actual, before) || !reflect.DeepEqual(input, before) {
				t.Fatalf("ambiguous reasoning data changed: %#v", actual)
			}
		})
	}
}

func TestReplayPreservesEmptyAndOpaqueReasoningDetails(t *testing.T) {
	for _, details := range []string{`[]`, `null`, `{"vendor":"opaque"}`, `[{"index":0,"type":"reasoning.text","text":"complete","signature":"signed"}]`} {
		data := json.RawMessage(`{"message":{"role":"assistant","content":"ok","reasoning_details":` + details + `}}`)
		message := agentcore.Message{Role: agentcore.RoleAssistant, ProviderData: &agentcore.ProviderData{Format: ProviderDataFormat, Data: data}}
		converted, err := convertMessage(message)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(converted, decodeJSONValue(data).(map[string]any)["message"]) {
			t.Fatalf("valid/opaque reasoning changed: %#v", converted)
		}
	}
	// Invalid opaque JSON still falls back to neutral content, as before.
	message := agentcore.TextMessage(agentcore.RoleAssistant, "neutral")
	message.ProviderData = &agentcore.ProviderData{Format: ProviderDataFormat, Data: json.RawMessage(`{"message":{"content":"opaque"}} {}`)}
	converted, err := convertMessage(message)
	if err != nil || converted["content"] != "neutral" {
		t.Fatalf("invalid JSON replayed: %#v, %v", converted, err)
	}
}

func TestModelReplaysLegacyReasoningDetailsWithoutMutatingHistory(t *testing.T) {
	const fragments = 6000
	var details []any
	for index := 0; index < fragments; index++ {
		details = append(details, map[string]any{"index": 0, "type": "reasoning.text", "text": "思考 ", "format": "test-v1"})
	}
	details = append(details,
		map[string]any{"index": 0, "signature": "signed"},
		map[string]any{"index": 1, "type": "reasoning.encrypted", "data": "YWJ"},
		map[string]any{"index": 1, "type": "reasoning.encrypted", "data": "j"},
	)
	receivedMessages := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		decoder := json.NewDecoder(request.Body)
		decoder.UseNumber()
		var body struct {
			Messages []map[string]any `json:"messages"`
		}
		if err := decoder.Decode(&body); err != nil || len(body.Messages) != 2 {
			http.Error(response, "invalid messages", http.StatusBadRequest)
			return
		}
		received := body.Messages[0]
		if len(received["reasoning_details"].([]any)) != 2 {
			http.Error(response, "fragmented reasoning history", http.StatusBadRequest)
			return
		}
		receivedMessages <- received
		response.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(response, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	model, err := New(Config{Model: "test", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	providerData, err := json.Marshal(preservedProviderData{Source: model.scope, Message: map[string]any{
		"role": "assistant", "content": nil, "reasoning_details": details,
		"provider_metadata": map[string]any{"request_id": json.Number("9007199254740993")},
		"tool_calls":        []any{map[string]any{"id": "call-1", "type": "function", "function": map[string]any{"name": "echo", "arguments": "{}"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	message := agentcore.Message{ID: "original", Role: agentcore.RoleAssistant, ProviderData: &agentcore.ProviderData{Format: ProviderDataFormat, Data: providerData}}
	// Exercise the same JSON round trip used by persisted sessions.
	serialized, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	var restored agentcore.Message
	if err = json.Unmarshal(serialized, &restored); err != nil {
		t.Fatal(err)
	}
	stream, err := model.Stream(context.Background(), agentcore.ModelRequest{Messages: []agentcore.Message{
		restored, {Role: agentcore.RoleTool, ToolCallID: "call-1", ToolName: "echo", Content: []agentcore.ContentBlock{{Type: agentcore.ContentText, Text: "tool result"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if chunk, err := stream.Recv(); err != nil || chunk.TextDelta != "ok" {
		t.Fatalf("chunk=%+v err=%v", chunk, err)
	}
	received := <-receivedMessages
	blocks := received["reasoning_details"].([]any)
	first := blocks[0].(map[string]any)
	if first["text"] != strings.Repeat("思考 ", fragments) || first["signature"] != "signed" || blocks[1].(map[string]any)["data"] != "YWJj" {
		t.Fatal("reasoning text, signature or encrypted payload changed")
	}
	if received["provider_metadata"].(map[string]any)["request_id"] != json.Number("9007199254740993") || received["tool_calls"].([]any)[0].(map[string]any)["id"] != "call-1" {
		t.Fatal("provider metadata or tool call changed")
	}
	if !bytes.Equal(providerData, restored.ProviderData.Data) || !bytes.Equal(providerData, message.ProviderData.Data) {
		t.Fatal("request conversion mutated persisted provider data")
	}
	// Normalized histories must remain unchanged on subsequent requests.
	normalized, _ := json.Marshal(preservedProviderData{Source: model.scope, Message: received})
	converted, err := convertMessageForScope(agentcore.Message{Role: agentcore.RoleAssistant, ProviderData: &agentcore.ProviderData{Format: ProviderDataFormat, Data: normalized}}, model.scope)
	if err != nil || !reflect.DeepEqual(converted, received) {
		t.Fatalf("replay is not idempotent: %v", err)
	}
	// Provider metadata stays scoped to the endpoint/model that produced it.
	other, err := convertMessageForScope(restored, "another-model")
	if err != nil || other["reasoning_details"] != nil {
		t.Fatalf("cross-model provider data replayed: %v", err)
	}
}
