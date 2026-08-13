package pimessages

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
	"sort"
	"strings"
	"sync"

	"github.com/z3r2ne/agentcore"
)

const ProviderDataFormat = "pi.messages.assistant.content/v1"

type Config struct {
	Model, APIKey, BaseURL                  string
	Headers                                 http.Header
	HTTPClient                              *http.Client
	MaxResponseBodyBytes, MaxErrorBodyBytes int64
	MaxSSEEventBytes                        int
}
type Model struct {
	model, endpoint, apiKey string
	headers                 http.Header
	client                  *http.Client
	maxBody, maxError       int64
	maxEvent                int
}

func New(config Config) (*Model, error) {
	if strings.TrimSpace(config.Model) == "" {
		return nil, errors.New("provider/pimessages: model is required")
	}
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = "https://radius.pi.dev"
	}
	u, err := url.Parse(base)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, errors.New("provider/pimessages: valid HTTP(S) BaseURL is required")
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/messages"
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
		return nil, errors.New("provider/pimessages: response limits must not be negative")
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Model{model: config.Model, endpoint: u.String(), apiKey: config.APIKey, headers: config.Headers.Clone(), client: client, maxBody: config.MaxResponseBodyBytes, maxError: config.MaxErrorBodyBytes, maxEvent: config.MaxSSEEventBytes}, nil
}
func (m *Model) Stream(ctx context.Context, request agentcore.ModelRequest) (agentcore.ModelStream, error) {
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
	if m.apiKey != "" {
		req.Header.Set("authorization", "Bearer "+m.apiKey)
	}
	for k, values := range m.headers {
		req.Header.Del(k)
		for _, v := range values {
			req.Header.Add(k, v)
		}
	}
	response, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("provider/pimessages: request: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		defer response.Body.Close()
		payload, _ := io.ReadAll(io.LimitReader(response.Body, m.maxError))
		return nil, fmt.Errorf("provider/pimessages: HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(payload)))
	}
	if !strings.HasPrefix(strings.ToLower(response.Header.Get("content-type")), "text/event-stream") {
		defer response.Body.Close()
		return nil, errors.New("provider/pimessages: expected text/event-stream")
	}
	return newStream(ctx, &limitedReader{body: response.Body, remaining: m.maxBody}, m.maxEvent), nil
}
func (m *Model) requestBody(request agentcore.ModelRequest) (map[string]any, error) {
	messages := make([]any, 0, len(request.Messages))
	for _, message := range request.Messages {
		converted, err := piMessage(message, m.model)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	tools := make([]any, len(request.Tools))
	for i, tool := range request.Tools {
		var parameters any = map[string]any{"type": "object", "properties": map[string]any{}}
		if len(tool.Parameters) > 0 {
			if err := json.Unmarshal(tool.Parameters, &parameters); err != nil {
				return nil, err
			}
		}
		tools[i] = map[string]any{"name": tool.Name, "description": tool.Description, "parameters": parameters}
	}
	options := map[string]any{}
	for k, v := range request.Options {
		options[k] = v
	}
	return map[string]any{"model": m.model, "context": map[string]any{"systemPrompt": request.SystemPrompt, "messages": messages, "tools": tools}, "options": options}, nil
}
func piMessage(message agentcore.Message, model string) (map[string]any, error) {
	switch message.Role {
	case agentcore.RoleUser, agentcore.RoleSystem:
		content := make([]any, 0, len(message.Content))
		for _, block := range message.Content {
			switch block.Type {
			case agentcore.ContentText:
				content = append(content, map[string]any{"type": "text", "text": block.Text})
			case agentcore.ContentImage:
				image := map[string]any{"type": "image", "mimeType": block.MIMEType}
				if len(block.Data) > 0 {
					image["data"] = base64.StdEncoding.EncodeToString(block.Data)
				} else {
					image["url"] = block.URL
				}
				content = append(content, image)
			}
		}
		return map[string]any{"role": "user", "content": content}, nil
	case agentcore.RoleAssistant:
		content := make([]any, 0, len(message.Content))
		if message.ProviderData != nil && message.ProviderData.Format == ProviderDataFormat && json.Valid(message.ProviderData.Data) {
			_ = json.Unmarshal(message.ProviderData.Data, &content)
		}
		if len(content) == 0 {
			for _, block := range message.Content {
				switch block.Type {
				case agentcore.ContentText:
					content = append(content, map[string]any{"type": "text", "text": block.Text})
				case agentcore.ContentThinking:
					content = append(content, map[string]any{"type": "thinking", "thinking": block.Text})
				case agentcore.ContentToolCall:
					if block.ToolCall != nil {
						var args any
						if err := json.Unmarshal(block.ToolCall.Arguments, &args); err != nil {
							return nil, err
						}
						content = append(content, map[string]any{"type": "toolCall", "id": block.ToolCall.ID, "name": block.ToolCall.Name, "arguments": args})
					}
				}
			}
		}
		usage := map[string]any{
			"input": message.Usage.InputTokens, "output": message.Usage.OutputTokens,
			"cacheRead": message.Usage.CacheReadTokens, "cacheWrite": message.Usage.CacheWriteTokens,
			"totalTokens": message.Usage.InputTokens + message.Usage.OutputTokens + message.Usage.CacheReadTokens + message.Usage.CacheWriteTokens,
			"cost":        map[string]any{"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
		}
		return map[string]any{"role": "assistant", "content": content, "api": "pi-messages", "provider": "radius", "model": model, "usage": usage, "stopReason": piStopOut(message.StopReason), "timestamp": 0}, nil
	case agentcore.RoleTool:
		content := make([]any, 0, len(message.Content))
		for _, block := range message.Content {
			switch block.Type {
			case agentcore.ContentText:
				content = append(content, map[string]any{"type": "text", "text": block.Text})
			case agentcore.ContentImage:
				image := map[string]any{"type": "image", "mimeType": block.MIMEType}
				if len(block.Data) > 0 {
					image["data"] = base64.StdEncoding.EncodeToString(block.Data)
				} else if block.URL != "" {
					image["url"] = block.URL
				} else {
					return nil, errors.New("provider/pimessages: tool image requires URL or data")
				}
				content = append(content, image)
			default:
				return nil, fmt.Errorf("provider/pimessages: unsupported tool content type %q", block.Type)
			}
		}
		return map[string]any{"role": "toolResult", "toolCallId": message.ToolCallID, "toolName": message.ToolName, "content": content, "isError": message.IsError, "timestamp": 0}, nil
	default:
		return nil, fmt.Errorf("provider/pimessages: unsupported role %q", message.Role)
	}
}
func piStopOut(reason agentcore.StopReason) string {
	switch reason {
	case agentcore.StopReasonToolUse:
		return "toolUse"
	case agentcore.StopReasonLength:
		return "length"
	case agentcore.StopReasonAborted:
		return "aborted"
	case agentcore.StopReasonError:
		return "error"
	default:
		return "stop"
	}
}

type limitedReader struct {
	body      io.ReadCloser
	remaining int64
	once      sync.Once
}

func (r *limitedReader) Read(p []byte) (int, error) {
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
func (r *limitedReader) Close() error {
	var e error
	r.once.Do(func() { e = r.body.Close() })
	return e
}

type stream struct {
	ctx              context.Context
	body             io.ReadCloser
	scanner          *bufio.Scanner
	max              int
	pending          []byte
	closed, terminal bool
	content          map[int]map[string]any
	toolArguments    map[int]string
}

func newStream(ctx context.Context, body io.ReadCloser, max int) *stream {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64<<10), max+1024)
	return &stream{ctx: ctx, body: body, scanner: scanner, max: max, content: map[int]map[string]any{}, toolArguments: map[int]string{}}
}
func (s *stream) Recv() (agentcore.ModelChunk, error) {
	if s.closed {
		return agentcore.ModelChunk{}, io.EOF
	}
	for {
		data, err := s.next()
		if err != nil {
			s.closed = true
			if errors.Is(err, io.EOF) && s.terminal {
				return agentcore.ModelChunk{}, io.EOF
			}
			if s.ctx.Err() != nil {
				return agentcore.ModelChunk{}, s.ctx.Err()
			}
			return agentcore.ModelChunk{}, io.ErrUnexpectedEOF
		}
		var event struct {
			Type         string `json:"type"`
			ContentIndex int    `json:"contentIndex"`
			Delta        string `json:"delta"`
			ID           string `json:"id"`
			ToolName     string `json:"toolName"`
			Reason       string `json:"reason"`
			ErrorMessage string `json:"errorMessage"`
			Content      string `json:"content"`
			Signature    string `json:"contentSignature"`
			Redacted     bool   `json:"redacted"`
			ToolCall     struct {
				ID, Name  string
				Arguments json.RawMessage
			} `json:"toolCall"`
			Usage struct{ Input, Output, CacheRead, CacheWrite int } `json:"usage"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			return agentcore.ModelChunk{}, err
		}
		chunk := agentcore.ModelChunk{}
		switch event.Type {
		case "start":
			continue
		case "text_start":
			s.content[event.ContentIndex] = map[string]any{"type": "text", "text": ""}
			continue
		case "text_delta":
			s.appendContent(event.ContentIndex, "text", event.Delta)
			chunk.TextDelta = event.Delta
			return chunk, nil
		case "text_end":
			s.finishContent(event.ContentIndex, "text", event.Content, event.Signature, false)
			continue
		case "thinking_start":
			s.content[event.ContentIndex] = map[string]any{"type": "thinking", "thinking": ""}
			continue
		case "thinking_delta":
			s.appendContent(event.ContentIndex, "thinking", event.Delta)
			chunk.ThinkingDelta = event.Delta
			return chunk, nil
		case "thinking_end":
			s.finishContent(event.ContentIndex, "thinking", event.Content, event.Signature, event.Redacted)
			continue
		case "toolcall_start":
			s.content[event.ContentIndex] = map[string]any{"type": "toolCall", "id": event.ID, "name": event.ToolName, "arguments": map[string]any{}}
			s.toolArguments[event.ContentIndex] = ""
			chunk.ToolCallDeltas = []agentcore.ToolCallDelta{{Index: event.ContentIndex, ID: event.ID, Name: event.ToolName}}
			return chunk, nil
		case "toolcall_delta":
			s.toolArguments[event.ContentIndex] += event.Delta
			chunk.ToolCallDeltas = []agentcore.ToolCallDelta{{Index: event.ContentIndex, ArgumentsDelta: event.Delta}}
			return chunk, nil
		case "toolcall_end":
			if block := s.content[event.ContentIndex]; block != nil {
				block["id"] = event.ToolCall.ID
				block["name"] = event.ToolCall.Name
				var arguments any
				if json.Unmarshal(event.ToolCall.Arguments, &arguments) != nil {
					_ = json.Unmarshal([]byte(s.toolArguments[event.ContentIndex]), &arguments)
				}
				block["arguments"] = arguments
			}
			delete(s.toolArguments, event.ContentIndex)
			continue
		case "done":
			s.terminal = true
			chunk.StopReason = piStop(event.Reason)
			usage := agentcore.Usage{InputTokens: event.Usage.Input, OutputTokens: event.Usage.Output, CacheReadTokens: event.Usage.CacheRead, CacheWriteTokens: event.Usage.CacheWrite}
			chunk.Usage = &usage
			data, marshalErr := json.Marshal(s.orderedContent())
			if marshalErr != nil {
				return chunk, marshalErr
			}
			chunk.ProviderData = &agentcore.ProviderData{Format: ProviderDataFormat, Data: data}
			return chunk, nil
		case "error":
			s.terminal = true
			s.closed = true
			if event.Reason == "aborted" {
				return chunk, context.Canceled
			}
			return chunk, fmt.Errorf("provider/pimessages: %s", event.ErrorMessage)
		default:
			continue
		}
	}
}

func (s *stream) appendContent(index int, field, delta string) {
	block := s.content[index]
	if block == nil {
		block = map[string]any{"type": field}
		s.content[index] = block
	}
	current, _ := block[field].(string)
	block[field] = current + delta
}

func (s *stream) finishContent(index int, field, content, signature string, redacted bool) {
	s.appendContent(index, field, "")
	block := s.content[index]
	block[field] = content
	if signature != "" {
		block[field+"Signature"] = signature
	}
	if redacted {
		block["redacted"] = true
	}
}

func (s *stream) orderedContent() []any {
	indexes := make([]int, 0, len(s.content))
	for index := range s.content {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	content := make([]any, 0, len(indexes))
	for _, index := range indexes {
		content = append(content, s.content[index])
	}
	return content
}
func (s *stream) next() ([]byte, error) {
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
func piStop(reason string) agentcore.StopReason {
	switch reason {
	case "toolUse":
		return agentcore.StopReasonToolUse
	case "length":
		return agentcore.StopReasonLength
	default:
		return agentcore.StopReasonStop
	}
}
func (s *stream) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.body.Close()
}

var _ agentcore.Model = (*Model)(nil)
var _ agentcore.ModelStream = (*stream)(nil)
