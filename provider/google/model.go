package google

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/z3r2ne/agentcore"
)

const ProviderDataFormat = "google.generate-content/v1"

// Config controls a Gemini API model.
type Config struct {
	Model                string
	APIKey               string
	BaseURL              string
	Headers              http.Header
	HTTPClient           *http.Client
	MaxResponseBodyBytes int64
	MaxErrorBodyBytes    int64
	MaxSSEEventBytes     int
}

// Model implements agentcore.Model using streamGenerateContent SSE.
type Model struct {
	endpoint, model, apiKey string
	headers                 http.Header
	client                  *http.Client
	maxBody, maxError       int64
	maxEvent                int
}

func New(config Config) (*Model, error) {
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, errors.New("provider/google: model is required")
	}
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = "https://generativelanguage.googleapis.com/v1beta"
	}
	u, err := url.Parse(base)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, errors.New("provider/google: valid HTTP(S) base URL is required")
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/models/" + url.PathEscape(model) + ":streamGenerateContent"
	query := u.Query()
	query.Set("alt", "sse")
	if config.APIKey != "" {
		query.Set("key", config.APIKey)
	}
	u.RawQuery = query.Encode()
	if config.MaxResponseBodyBytes == 0 {
		config.MaxResponseBodyBytes = 64 << 20
	}
	if config.MaxErrorBodyBytes == 0 {
		config.MaxErrorBodyBytes = 1 << 20
	}
	if config.MaxSSEEventBytes == 0 {
		config.MaxSSEEventBytes = 4 << 20
	}
	if config.MaxResponseBodyBytes < 0 || config.MaxErrorBodyBytes < 0 || config.MaxSSEEventBytes < 0 {
		return nil, errors.New("provider/google: response limits must not be negative")
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Model{endpoint: u.String(), model: model, apiKey: config.APIKey, headers: config.Headers.Clone(), client: client, maxBody: config.MaxResponseBodyBytes, maxError: config.MaxErrorBodyBytes, maxEvent: config.MaxSSEEventBytes}, nil
}

func (m *Model) Stream(ctx context.Context, request agentcore.ModelRequest) (agentcore.ModelStream, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	body, err := m.requestBody(request)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.endpoint, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("accept", "text/event-stream")
	for k, values := range m.headers {
		req.Header.Del(k)
		for _, v := range values {
			req.Header.Add(k, v)
		}
	}
	response, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("provider/google: request: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		defer response.Body.Close()
		payload, _ := io.ReadAll(io.LimitReader(response.Body, m.maxError))
		return nil, fmt.Errorf("provider/google: HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(payload)))
	}
	if !strings.HasPrefix(strings.ToLower(response.Header.Get("content-type")), "text/event-stream") {
		defer response.Body.Close()
		return nil, errors.New("provider/google: expected text/event-stream")
	}
	return newGoogleStream(ctx, &googleLimitedReader{body: response.Body, remaining: m.maxBody}, m.maxEvent), nil
}

func (m *Model) requestBody(request agentcore.ModelRequest) (map[string]any, error) {
	contents := make([]map[string]any, 0, len(request.Messages))
	for _, message := range request.Messages {
		converted, err := googleMessage(message)
		if err != nil {
			return nil, err
		}
		if len(contents) > 0 && contents[len(contents)-1]["role"] == converted["role"] {
			previous, _ := contents[len(contents)-1]["parts"].([]any)
			next, _ := converted["parts"].([]any)
			contents[len(contents)-1]["parts"] = append(previous, next...)
		} else {
			contents = append(contents, converted)
		}
	}
	body := map[string]any{"contents": contents}
	if request.SystemPrompt != "" {
		body["systemInstruction"] = map[string]any{"parts": []any{map[string]any{"text": request.SystemPrompt}}}
	}
	if len(request.Tools) > 0 {
		declarations := make([]any, len(request.Tools))
		for i, tool := range request.Tools {
			var parameters any = map[string]any{"type": "object", "properties": map[string]any{}}
			if len(tool.Parameters) > 0 {
				if err := json.Unmarshal(tool.Parameters, &parameters); err != nil {
					return nil, err
				}
			}
			declarations[i] = map[string]any{"name": tool.Name, "description": tool.Description, "parameters": parameters}
		}
		body["tools"] = []any{map[string]any{"functionDeclarations": declarations}}
	}
	generation := map[string]any{}
	for key, value := range request.Options {
		switch key {
		case "contents", "systemInstruction", "tools":
			continue
		case "safetySettings", "toolConfig", "cachedContent", "labels":
			body[key] = value
		default:
			generation[key] = value
		}
	}
	if len(generation) > 0 {
		body["generationConfig"] = generation
	}
	return body, nil
}

func googleMessage(message agentcore.Message) (map[string]any, error) {
	if message.Role == agentcore.RoleAssistant && message.ProviderData != nil && message.ProviderData.Format == ProviderDataFormat && json.Valid(message.ProviderData.Data) {
		var content map[string]any
		if json.Unmarshal(message.ProviderData.Data, &content) == nil {
			return content, nil
		}
	}
	role := "user"
	if message.Role == agentcore.RoleAssistant {
		role = "model"
	}
	parts := make([]any, 0, len(message.Content))
	switch message.Role {
	case agentcore.RoleSystem, agentcore.RoleUser, agentcore.RoleAssistant:
		for _, block := range message.Content {
			switch block.Type {
			case agentcore.ContentText:
				parts = append(parts, map[string]any{"text": block.Text})
			case agentcore.ContentThinking:
				parts = append(parts, map[string]any{"text": block.Text, "thought": true})
			case agentcore.ContentToolCall:
				if block.ToolCall != nil {
					var args any
					if err := json.Unmarshal(block.ToolCall.Arguments, &args); err != nil {
						return nil, err
					}
					parts = append(parts, map[string]any{"functionCall": map[string]any{"name": block.ToolCall.Name, "args": args, "id": block.ToolCall.ID}})
				}
			case agentcore.ContentImage, agentcore.ContentAudio, agentcore.ContentVideo, agentcore.ContentFile:
				if len(block.Data) > 0 && block.MIMEType != "" {
					parts = append(parts, map[string]any{"inlineData": map[string]any{"mimeType": block.MIMEType, "data": base64.StdEncoding.EncodeToString(block.Data)}})
				} else if block.URL != "" {
					parts = append(parts, map[string]any{"fileData": map[string]any{"mimeType": block.MIMEType, "fileUri": block.URL}})
				} else {
					return nil, errors.New("provider/google: media requires URL or MIME-typed data")
				}
			}
		}
	case agentcore.RoleTool:
		var response any = message.Text()
		parts = append(parts, map[string]any{"functionResponse": map[string]any{"name": message.ToolName, "id": message.ToolCallID, "response": map[string]any{"output": response, "isError": message.IsError}}})
	default:
		return nil, fmt.Errorf("provider/google: unsupported role %q", message.Role)
	}
	return map[string]any{"role": role, "parts": parts}, nil
}

type googleLimitedReader struct {
	body      io.ReadCloser
	remaining int64
	once      sync.Once
}

func (r *googleLimitedReader) Read(p []byte) (int, error) {
	if r.remaining <= 0 {
		return 0, errors.New("response body exceeds configured limit")
	}
	if int64(len(p)) > r.remaining {
		p = p[:r.remaining]
	}
	n, e := r.body.Read(p)
	r.remaining -= int64(n)
	return n, e
}
func (r *googleLimitedReader) Close() error {
	var e error
	r.once.Do(func() { e = r.body.Close() })
	return e
}

type googleStream struct {
	ctx                     context.Context
	body                    io.ReadCloser
	scanner                 *bufio.Scanner
	max                     int
	pending                 []byte
	done, closed, sawFinish bool
	partIndex               int
	content                 map[string]any
}

func newGoogleStream(ctx context.Context, body io.ReadCloser, max int) *googleStream {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64<<10), max+1024)
	return &googleStream{ctx: ctx, body: body, scanner: scanner, max: max, content: map[string]any{"role": "model", "parts": []any{}}}
}
func (s *googleStream) Recv() (agentcore.ModelChunk, error) {
	if s.done || s.closed {
		return agentcore.ModelChunk{}, io.EOF
	}
	payload, err := s.next()
	if err != nil {
		s.done = true
		if errors.Is(err, io.EOF) && s.sawFinish {
			return agentcore.ModelChunk{}, io.EOF
		}
		if s.ctx.Err() != nil {
			return agentcore.ModelChunk{}, s.ctx.Err()
		}
		return agentcore.ModelChunk{}, fmt.Errorf("provider/google: stream ended without finishReason: %w", io.ErrUnexpectedEOF)
	}
	var envelope struct {
		Candidates []struct {
			Content struct {
				Role  string `json:"role"`
				Parts []struct {
					Text             string `json:"text"`
					Thought          bool   `json:"thought"`
					ThoughtSignature string `json:"thoughtSignature"`
					FunctionCall     *struct {
						Name string          `json:"name"`
						ID   string          `json:"id"`
						Args json.RawMessage `json:"args"`
					} `json:"functionCall"`
				} `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		Usage struct {
			Prompt, Candidates, Cached int `json:"-"`
			PromptTokens               int `json:"promptTokenCount"`
			CandidateTokens            int `json:"candidatesTokenCount"`
			CachedTokens               int `json:"cachedContentTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return agentcore.ModelChunk{}, fmt.Errorf("provider/google: decode stream: %w", err)
	}
	chunk := agentcore.ModelChunk{}
	if len(envelope.Candidates) > 0 {
		candidate := envelope.Candidates[0]
		parts := s.content["parts"].([]any)
		for _, part := range candidate.Content.Parts {
			wire := map[string]any{}
			if part.FunctionCall != nil {
				var args any
				if len(part.FunctionCall.Args) > 0 {
					_ = json.Unmarshal(part.FunctionCall.Args, &args)
				}
				wire["functionCall"] = map[string]any{"name": part.FunctionCall.Name, "id": part.FunctionCall.ID, "args": args}
				arguments, _ := json.Marshal(args)
				chunk.ToolCallDeltas = append(chunk.ToolCallDeltas, agentcore.ToolCallDelta{Index: s.partIndex, ID: part.FunctionCall.ID, Name: part.FunctionCall.Name, ArgumentsDelta: string(arguments)})
			} else {
				wire["text"] = part.Text
				if part.Thought {
					wire["thought"] = true
					chunk.ThinkingDelta += part.Text
				} else {
					chunk.TextDelta += part.Text
				}
				if part.ThoughtSignature != "" {
					wire["thoughtSignature"] = part.ThoughtSignature
				}
			}
			parts = append(parts, wire)
			s.partIndex++
		}
		s.content["parts"] = parts
		if candidate.FinishReason != "" {
			reason, finishErr := googleStop(candidate.FinishReason)
			if finishErr != nil {
				return agentcore.ModelChunk{}, finishErr
			}
			chunk.StopReason = reason
			if len(chunk.ToolCallDeltas) > 0 && reason == agentcore.StopReasonStop {
				chunk.StopReason = agentcore.StopReasonToolUse
			}
			s.sawFinish = true
			s.done = true
		}
	}
	if envelope.Usage.PromptTokens != 0 || envelope.Usage.CandidateTokens != 0 {
		usage := agentcore.Usage{InputTokens: max(0, envelope.Usage.PromptTokens-envelope.Usage.CachedTokens), OutputTokens: envelope.Usage.CandidateTokens, CacheReadTokens: envelope.Usage.CachedTokens}
		chunk.Usage = &usage
	}
	if chunk.StopReason != "" {
		data, _ := json.Marshal(s.content)
		chunk.ProviderData = &agentcore.ProviderData{Format: ProviderDataFormat, Data: data}
	}
	return chunk, nil
}
func (s *googleStream) next() ([]byte, error) {
	for s.scanner.Scan() {
		line := s.scanner.Bytes()
		if len(line) == 0 {
			if len(s.pending) == 0 {
				continue
			}
			data := append([]byte(nil), s.pending...)
			s.pending = nil
			return data, nil
		}
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		value := bytes.TrimSpace(line[5:])
		if len(s.pending) > 0 {
			s.pending = append(s.pending, '\n')
		}
		if len(s.pending)+len(value) > s.max {
			return nil, errors.New("SSE event exceeds configured limit")
		}
		s.pending = append(s.pending, value...)
	}
	if err := s.scanner.Err(); err != nil {
		return nil, err
	}
	return nil, io.EOF
}
func googleStop(reason string) (agentcore.StopReason, error) {
	switch strings.ToUpper(reason) {
	case "STOP":
		return agentcore.StopReasonStop, nil
	case "MAX_TOKENS":
		return agentcore.StopReasonLength, nil
	case "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL":
		return agentcore.StopReasonError, fmt.Errorf("provider/google: finishReason %s", reason)
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY":
		return agentcore.StopReasonError, fmt.Errorf("provider/google: blocked generation: %s", reason)
	default:
		return agentcore.StopReasonError, fmt.Errorf("provider/google: unsupported finishReason %s", reason)
	}
}
func (s *googleStream) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.body.Close()
}

var _ agentcore.Model = (*Model)(nil)
var _ agentcore.ModelStream = (*googleStream)(nil)
