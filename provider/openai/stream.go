package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/z3r2ne/agentcore"
)

var errResponseTooLarge = errors.New("response body exceeds configured limit")

type limitReadCloser struct {
	body      io.ReadCloser
	remaining int64
	closeOnce sync.Once
	closeErr  error
}

func newLimitReadCloser(body io.ReadCloser, limit int64) *limitReadCloser {
	return &limitReadCloser{body: body, remaining: limit}
}

func (r *limitReadCloser) Read(buffer []byte) (int, error) {
	if r.remaining == 0 {
		var probe [1]byte
		n, err := r.body.Read(probe[:])
		if n > 0 {
			return 0, errResponseTooLarge
		}
		return 0, err
	}
	if int64(len(buffer)) > r.remaining {
		buffer = buffer[:r.remaining]
	}
	n, err := r.body.Read(buffer)
	r.remaining -= int64(n)
	return n, err
}

func (r *limitReadCloser) Close() error {
	r.closeOnce.Do(func() { r.closeErr = r.body.Close() })
	return r.closeErr
}

type stream struct {
	ctx          context.Context
	body         io.ReadCloser
	scanner      *bufio.Scanner
	maxEventSize int
	pendingData  []byte
	done         bool
	closed       atomic.Bool

	message   map[string]any
	toolCalls map[int]map[string]any
	response  map[string]any
}

func newStream(ctx context.Context, body io.ReadCloser, maxEventSize int) *stream {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), maxEventSize+1024)
	return &stream{
		ctx: ctx, body: body, scanner: scanner, maxEventSize: maxEventSize,
		message: map[string]any{"role": "assistant"}, toolCalls: make(map[int]map[string]any),
		response: make(map[string]any),
	}
}

type streamEnvelope struct {
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           json.Number     `json:"created"`
	Model             string          `json:"model"`
	ServiceTier       string          `json:"service_tier"`
	SystemFingerprint string          `json:"system_fingerprint"`
	Choices           []streamChoice  `json:"choices"`
	Usage             json.RawMessage `json:"usage"`
	Error             *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error"`
}

type streamChoice struct {
	Index        int             `json:"index"`
	Delta        json.RawMessage `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type typedDelta struct {
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content"`
	ToolCalls        []struct {
		Index    int    `json:"index"`
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

func (s *stream) Recv() (agentcore.ModelChunk, error) {
	if s.done || s.closed.Load() {
		return agentcore.ModelChunk{}, io.EOF
	}
	for {
		payload, err := s.nextEvent()
		if err != nil {
			s.done = true
			if errors.Is(err, io.EOF) {
				return agentcore.ModelChunk{}, io.EOF
			}
			if s.ctx.Err() != nil {
				return agentcore.ModelChunk{}, s.ctx.Err()
			}
			return agentcore.ModelChunk{}, &Error{Operation: "read stream", Retryable: true, Err: err}
		}
		if bytes.Equal(bytes.TrimSpace(payload), []byte("[DONE]")) {
			s.done = true
			return agentcore.ModelChunk{}, io.EOF
		}
		chunk, skip, err := s.decodeEvent(payload)
		if err != nil {
			s.done = true
			return agentcore.ModelChunk{}, err
		}
		if skip {
			continue
		}
		return chunk, nil
	}
}

func (s *stream) nextEvent() ([]byte, error) {
	for s.scanner.Scan() {
		line := s.scanner.Bytes()
		if len(line) == 0 {
			if len(s.pendingData) == 0 {
				continue
			}
			payload := append([]byte(nil), s.pendingData...)
			s.pendingData = s.pendingData[:0]
			return payload, nil
		}
		if line[0] == ':' {
			continue
		}
		field, value, found := bytes.Cut(line, []byte{':'})
		if !found || !bytes.Equal(field, []byte("data")) {
			continue
		}
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}
		if len(s.pendingData) > 0 {
			s.pendingData = append(s.pendingData, '\n')
		}
		if len(s.pendingData)+len(value) > s.maxEventSize {
			return nil, fmt.Errorf("SSE event exceeds %d bytes", s.maxEventSize)
		}
		s.pendingData = append(s.pendingData, value...)
	}
	if err := s.scanner.Err(); err != nil {
		return nil, err
	}
	if len(s.pendingData) > 0 {
		payload := append([]byte(nil), s.pendingData...)
		s.pendingData = nil
		return payload, nil
	}
	return nil, io.EOF
}

func (s *stream) decodeEvent(payload []byte) (agentcore.ModelChunk, bool, error) {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var envelope streamEnvelope
	if err := decoder.Decode(&envelope); err != nil {
		return agentcore.ModelChunk{}, false, &Error{Operation: "decode stream event", Retryable: true, Err: err}
	}
	if envelope.Error != nil {
		providerError := &Error{Operation: "stream", Body: envelope.Error.Message, Type: envelope.Error.Type, Retryable: true}
		if envelope.Error.Code != nil {
			providerError.Code = fmt.Sprint(envelope.Error.Code)
		}
		return agentcore.ModelChunk{}, false, providerError
	}
	s.captureResponse(envelope)

	chunk := agentcore.ModelChunk{}
	if len(envelope.Usage) > 0 && !bytes.Equal(bytes.TrimSpace(envelope.Usage), []byte("null")) {
		usage, err := decodeUsage(envelope.Usage)
		if err != nil {
			return agentcore.ModelChunk{}, false, &Error{Operation: "decode usage", Retryable: true, Err: err}
		}
		chunk.Usage = &usage
		s.response["usage"] = decodeJSONValue(envelope.Usage)
	}
	choice, ok := primaryChoice(envelope.Choices)
	if ok {
		var delta typedDelta
		if len(choice.Delta) > 0 {
			if err := json.Unmarshal(choice.Delta, &delta); err != nil {
				return agentcore.ModelChunk{}, false, &Error{Operation: "decode choice delta", Retryable: true, Err: err}
			}
			s.mergeProviderDelta(choice.Delta)
		}
		chunk.TextDelta = delta.Content
		chunk.ThinkingDelta = delta.ReasoningContent
		chunk.ToolCallDeltas = make([]agentcore.ToolCallDelta, len(delta.ToolCalls))
		for index, call := range delta.ToolCalls {
			chunk.ToolCallDeltas[index] = agentcore.ToolCallDelta{
				Index: call.Index, ID: call.ID, Name: call.Function.Name, ArgumentsDelta: call.Function.Arguments,
			}
		}
		if choice.FinishReason != nil {
			chunk.StopReason = stopReason(*choice.FinishReason)
		}
	}
	providerData, err := s.providerData()
	if err != nil {
		return agentcore.ModelChunk{}, false, &Error{Operation: "encode provider data", Retryable: false, Err: err}
	}
	chunk.ProviderData = providerData
	return chunk, !ok && chunk.Usage == nil, nil
}

func primaryChoice(choices []streamChoice) (streamChoice, bool) {
	for _, choice := range choices {
		if choice.Index == 0 {
			return choice, true
		}
	}
	if len(choices) > 0 {
		return choices[0], true
	}
	return streamChoice{}, false
}

func (s *stream) captureResponse(envelope streamEnvelope) {
	if envelope.ID != "" {
		s.response["id"] = envelope.ID
	}
	if envelope.Object != "" {
		s.response["object"] = envelope.Object
	}
	if envelope.Created != "" {
		s.response["created"] = envelope.Created
	}
	if envelope.Model != "" {
		s.response["model"] = envelope.Model
	}
	if envelope.ServiceTier != "" {
		s.response["service_tier"] = envelope.ServiceTier
	}
	if envelope.SystemFingerprint != "" {
		s.response["system_fingerprint"] = envelope.SystemFingerprint
	}
}

func (s *stream) mergeProviderDelta(raw json.RawMessage) {
	value := decodeJSONValue(raw)
	delta, ok := value.(map[string]any)
	if !ok {
		return
	}
	if calls, ok := delta["tool_calls"].([]any); ok {
		for _, rawCall := range calls {
			call, ok := rawCall.(map[string]any)
			if !ok {
				continue
			}
			index := jsonInt(call["index"])
			delete(call, "index")
			target := s.toolCalls[index]
			if target == nil {
				target = make(map[string]any)
				s.toolCalls[index] = target
			}
			mergeObject(target, call, "")
		}
	}
	delete(delta, "tool_calls")
	mergeObject(s.message, delta, "")
}

func mergeObject(target, delta map[string]any, parent string) {
	for key, value := range delta {
		path := key
		if parent != "" {
			path = parent + "." + key
		}
		switch typed := value.(type) {
		case string:
			if path == "content" || path == "reasoning_content" || path == "refusal" || path == "function.arguments" || path == "function.name" {
				if existing, ok := target[key].(string); ok {
					target[key] = existing + typed
				} else {
					target[key] = typed
				}
			} else if typed != "" {
				target[key] = typed
			}
		case map[string]any:
			nested, _ := target[key].(map[string]any)
			if nested == nil {
				nested = make(map[string]any)
				target[key] = nested
			}
			mergeObject(nested, typed, path)
		case []any:
			existing, _ := target[key].([]any)
			target[key] = append(existing, cloneJSONSlice(typed)...)
		case nil:
			if _, exists := target[key]; !exists {
				target[key] = nil
			}
		default:
			target[key] = typed
		}
	}
}

func cloneJSONSlice(input []any) []any {
	result := make([]any, len(input))
	for index := range input {
		result[index] = cloneJSONValue(input[index])
	}
	return result
}

func jsonInt(value any) int {
	switch typed := value.(type) {
	case json.Number:
		result, _ := typed.Int64()
		return int(result)
	case float64:
		return int(typed)
	case int:
		return typed
	default:
		return 0
	}
}

type preservedProviderData struct {
	Message  map[string]any `json:"message"`
	Response map[string]any `json:"response,omitempty"`
}

func (s *stream) providerData() (*agentcore.ProviderData, error) {
	message := cloneMap(s.message)
	if len(s.toolCalls) > 0 {
		indexes := make([]int, 0, len(s.toolCalls))
		for index := range s.toolCalls {
			indexes = append(indexes, index)
		}
		sort.Ints(indexes)
		calls := make([]any, 0, len(indexes))
		for _, index := range indexes {
			calls = append(calls, cloneMap(s.toolCalls[index]))
		}
		message["tool_calls"] = calls
	}
	preserved := preservedProviderData{Message: message}
	if len(s.response) > 0 {
		preserved.Response = cloneMap(s.response)
	}
	data, err := json.Marshal(preserved)
	if err != nil {
		return nil, err
	}
	return &agentcore.ProviderData{Format: ProviderDataFormat, Data: data}, nil
}

func decodeJSONValue(raw json.RawMessage) any {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) != nil {
		return nil
	}
	return value
}

func decodeUsage(raw json.RawMessage) (agentcore.Usage, error) {
	var usage struct {
		PromptTokens             int `json:"prompt_tokens"`
		CompletionTokens         int `json:"completion_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		PromptCacheHitTokens     int `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens    int `json:"prompt_cache_miss_tokens"`
		PromptTokensDetails      struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	}
	if err := json.Unmarshal(raw, &usage); err != nil {
		return agentcore.Usage{}, err
	}
	cacheRead := usage.PromptTokensDetails.CachedTokens
	if usage.CacheReadInputTokens != 0 {
		cacheRead = usage.CacheReadInputTokens
	} else if usage.PromptCacheHitTokens != 0 {
		cacheRead = usage.PromptCacheHitTokens
	}
	cacheWrite := usage.CacheCreationInputTokens
	if cacheWrite == 0 {
		cacheWrite = usage.PromptCacheMissTokens
	}
	return agentcore.Usage{
		InputTokens: usage.PromptTokens, OutputTokens: usage.CompletionTokens,
		CacheReadTokens: cacheRead, CacheWriteTokens: cacheWrite,
	}, nil
}

func (s *stream) Close() error {
	if s == nil || s.body == nil || s.closed.Swap(true) {
		return nil
	}
	return s.body.Close()
}

func stopReason(reason string) agentcore.StopReason {
	switch strings.ToLower(strings.TrimSpace(reason)) {
	case "stop", "end_turn":
		return agentcore.StopReasonStop
	case "tool_calls", "function_call", "tool_use":
		return agentcore.StopReasonToolUse
	case "length", "max_tokens", "max_output_tokens":
		return agentcore.StopReasonLength
	case "cancelled", "canceled":
		return agentcore.StopReasonAborted
	case "content_filter", "error":
		return agentcore.StopReasonError
	case "":
		return ""
	default:
		return agentcore.StopReason(reason)
	}
}
