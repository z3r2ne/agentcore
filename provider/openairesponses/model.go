package openairesponses

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

const ProviderDataFormat = "openai.responses.output/v1"

// Config controls an OpenAI Responses API model. It also supports compatible
// Responses endpoints through BaseURL and custom headers.
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
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, errors.New("provider/openairesponses: model is required")
	}
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = "https://api.openai.com/v1"
	}
	u, err := url.Parse(base)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, errors.New("provider/openairesponses: valid HTTP(S) base URL is required")
	}
	u.Path = strings.TrimRight(u.Path, "/")
	if !strings.HasSuffix(u.Path, "/responses") {
		u.Path += "/responses"
	}
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
		return nil, errors.New("provider/openairesponses: response limits must not be negative")
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Model{model: model, endpoint: u.String(), apiKey: strings.TrimSpace(config.APIKey), headers: config.Headers.Clone(), client: client, maxBody: config.MaxResponseBodyBytes, maxError: config.MaxErrorBodyBytes, maxEvent: config.MaxSSEEventBytes}, nil
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
		return nil, fmt.Errorf("provider/openairesponses: request: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		defer response.Body.Close()
		payload, _ := io.ReadAll(io.LimitReader(response.Body, m.maxError))
		return nil, fmt.Errorf("provider/openairesponses: HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(payload)))
	}
	if !strings.HasPrefix(strings.ToLower(response.Header.Get("content-type")), "text/event-stream") {
		defer response.Body.Close()
		return nil, errors.New("provider/openairesponses: expected text/event-stream")
	}
	return newResponseStream(ctx, &responseLimitedReader{body: response.Body, remaining: m.maxBody}, m.maxEvent), nil
}

func (m *Model) requestBody(request agentcore.ModelRequest) (map[string]any, error) {
	input := make([]any, 0, len(request.Messages))
	for _, message := range request.Messages {
		items, err := responseItems(message)
		if err != nil {
			return nil, err
		}
		input = append(input, items...)
	}
	body := map[string]any{"model": m.model, "input": input, "stream": true, "store": false}
	if request.SystemPrompt != "" {
		body["instructions"] = request.SystemPrompt
	}
	if len(request.Tools) > 0 {
		tools := make([]any, len(request.Tools))
		for i, tool := range request.Tools {
			var parameters any = map[string]any{"type": "object", "properties": map[string]any{}}
			if len(tool.Parameters) > 0 {
				if err := json.Unmarshal(tool.Parameters, &parameters); err != nil {
					return nil, err
				}
			}
			tools[i] = map[string]any{"type": "function", "name": tool.Name, "description": tool.Description, "parameters": parameters}
		}
		body["tools"] = tools
	}
	for key, value := range request.Options {
		switch key {
		case "model", "input", "stream", "store", "instructions", "tools":
		default:
			body[key] = value
		}
	}
	return body, nil
}

func responseItems(message agentcore.Message) ([]any, error) {
	if message.Role == agentcore.RoleAssistant && message.ProviderData != nil && message.ProviderData.Format == ProviderDataFormat && json.Valid(message.ProviderData.Data) {
		var items []any
		if json.Unmarshal(message.ProviderData.Data, &items) == nil {
			return items, nil
		}
	}
	switch message.Role {
	case agentcore.RoleTool:
		return []any{map[string]any{"type": "function_call_output", "call_id": message.ToolCallID, "output": message.Text()}}, nil
	case agentcore.RoleSystem, agentcore.RoleUser, agentcore.RoleAssistant:
		role := string(message.Role)
		items := make([]any, 0)
		content := make([]any, 0)
		for _, block := range message.Content {
			switch block.Type {
			case agentcore.ContentText:
				kind := "input_text"
				if message.Role == agentcore.RoleAssistant {
					kind = "output_text"
				}
				content = append(content, map[string]any{"type": kind, "text": block.Text})
			case agentcore.ContentImage:
				if message.Role == agentcore.RoleAssistant {
					continue
				}
				imageURL := block.URL
				if imageURL == "" && len(block.Data) > 0 && block.MIMEType != "" {
					imageURL = "data:" + block.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(block.Data)
				}
				if imageURL == "" {
					return nil, errors.New("provider/openairesponses: image requires URL or MIME-typed data")
				}
				content = append(content, map[string]any{"type": "input_image", "image_url": imageURL})
			case agentcore.ContentFile:
				if message.Role == agentcore.RoleAssistant {
					continue
				}
				part := map[string]any{"type": "input_file"}
				if block.URL != "" {
					part["file_url"] = block.URL
				} else if len(block.Data) > 0 {
					part["file_data"] = "data:" + block.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(block.Data)
				}
				if block.Name != "" {
					part["filename"] = block.Name
				}
				content = append(content, part)
			case agentcore.ContentToolCall:
				if block.ToolCall != nil {
					items = append(items, map[string]any{"type": "function_call", "call_id": block.ToolCall.ID, "name": block.ToolCall.Name, "arguments": string(block.ToolCall.Arguments)})
				}
			}
		}
		if len(content) > 0 {
			items = append([]any{map[string]any{"type": "message", "role": role, "content": content}}, items...)
		}
		return items, nil
	default:
		return nil, fmt.Errorf("provider/openairesponses: unsupported role %q", message.Role)
	}
}

type responseLimitedReader struct {
	body      io.ReadCloser
	remaining int64
	once      sync.Once
}

func (r *responseLimitedReader) Read(p []byte) (int, error) {
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
func (r *responseLimitedReader) Close() error {
	var e error
	r.once.Do(func() { e = r.body.Close() })
	return e
}

type responseStream struct {
	ctx                       context.Context
	body                      io.ReadCloser
	scanner                   *bufio.Scanner
	max                       int
	event                     string
	pending                   []byte
	done, closed, sawTerminal bool
	items                     map[int]map[string]any
	callIndex                 map[string]int
	nextIndex                 int
}

func newResponseStream(ctx context.Context, body io.ReadCloser, max int) *responseStream {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64<<10), max+1024)
	return &responseStream{ctx: ctx, body: body, scanner: scanner, max: max, items: map[int]map[string]any{}, callIndex: map[string]int{}}
}
func (s *responseStream) Recv() (agentcore.ModelChunk, error) {
	if s.done || s.closed {
		return agentcore.ModelChunk{}, io.EOF
	}
	for {
		event, data, err := s.next()
		if err != nil {
			s.done = true
			if errors.Is(err, io.EOF) && s.sawTerminal {
				return agentcore.ModelChunk{}, io.EOF
			}
			if s.ctx.Err() != nil {
				return agentcore.ModelChunk{}, s.ctx.Err()
			}
			return agentcore.ModelChunk{}, fmt.Errorf("provider/openairesponses: stream ended before terminal event: %w", io.ErrUnexpectedEOF)
		}
		chunk, skip, err := s.decode(event, data)
		if err != nil {
			s.done = true
			return agentcore.ModelChunk{}, err
		}
		if !skip {
			return chunk, nil
		}
	}
}
func (s *responseStream) next() (string, []byte, error) {
	for s.scanner.Scan() {
		line := s.scanner.Bytes()
		if len(line) == 0 {
			if len(s.pending) == 0 {
				continue
			}
			event := s.event
			data := append([]byte(nil), s.pending...)
			s.event = ""
			s.pending = nil
			return event, data, nil
		}
		field, value, ok := bytes.Cut(line, []byte(":"))
		if !ok {
			continue
		}
		value = bytes.TrimSpace(value)
		switch string(field) {
		case "event":
			s.event = string(value)
		case "data":
			if len(s.pending) > 0 {
				s.pending = append(s.pending, '\n')
			}
			if len(s.pending)+len(value) > s.max {
				return "", nil, errors.New("SSE event exceeds configured limit")
			}
			s.pending = append(s.pending, value...)
		}
	}
	if err := s.scanner.Err(); err != nil {
		return "", nil, err
	}
	return "", nil, io.EOF
}
func (s *responseStream) decode(event string, data []byte) (agentcore.ModelChunk, bool, error) {
	var base struct {
		Type        string                         `json:"type"`
		OutputIndex int                            `json:"output_index"`
		ItemID      string                         `json:"item_id"`
		Delta       string                         `json:"delta"`
		Item        map[string]any                 `json:"item"`
		Response    json.RawMessage                `json:"response"`
		Error       struct{ Message, Code string } `json:"error"`
	}
	if err := json.Unmarshal(data, &base); err != nil {
		return agentcore.ModelChunk{}, false, fmt.Errorf("provider/openairesponses: decode event: %w", err)
	}
	kind := base.Type
	if kind == "" {
		kind = event
	}
	chunk := agentcore.ModelChunk{}
	switch kind {
	case "response.output_text.delta":
		chunk.TextDelta = base.Delta
		return chunk, false, nil
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		chunk.ThinkingDelta = base.Delta
		return chunk, false, nil
	case "response.output_item.added":
		if base.Item != nil {
			s.items[base.OutputIndex] = base.Item
			if itemType, _ := base.Item["type"].(string); itemType == "function_call" {
				callID, _ := base.Item["call_id"].(string)
				name, _ := base.Item["name"].(string)
				index := s.indexFor(callID, base.OutputIndex)
				chunk.ToolCallDeltas = []agentcore.ToolCallDelta{{Index: index, ID: callID, Name: name}}
				return chunk, false, nil
			}
		}
		return chunk, true, nil
	case "response.function_call_arguments.delta":
		index := s.indexFor(base.ItemID, base.OutputIndex)
		if item := s.items[base.OutputIndex]; item != nil {
			existing, _ := item["arguments"].(string)
			item["arguments"] = existing + base.Delta
		}
		chunk.ToolCallDeltas = []agentcore.ToolCallDelta{{Index: index, ArgumentsDelta: base.Delta}}
		return chunk, false, nil
	case "response.output_item.done":
		if base.Item != nil {
			s.items[base.OutputIndex] = base.Item
		}
		return chunk, true, nil
	case "response.completed", "response.incomplete":
		var response struct {
			Output            []any  `json:"output"`
			Status            string `json:"status"`
			IncompleteDetails struct {
				Reason string `json:"reason"`
			} `json:"incomplete_details"`
			Usage struct {
				Input, Output int `json:"-"`
				InputTokens   int `json:"input_tokens"`
				OutputTokens  int `json:"output_tokens"`
				Details       struct {
					Cached int `json:"cached_tokens"`
				} `json:"input_tokens_details"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(base.Response, &response); err != nil {
			return chunk, false, err
		}
		output := response.Output
		if len(output) == 0 {
			output = s.orderedItems()
		}
		hasCall := false
		for _, raw := range output {
			if item, ok := raw.(map[string]any); ok && item["type"] == "function_call" {
				hasCall = true
			}
		}
		if hasCall {
			chunk.StopReason = agentcore.StopReasonToolUse
		} else {
			chunk.StopReason = agentcore.StopReasonStop
		}
		if kind == "response.incomplete" || response.Status == "incomplete" {
			if response.IncompleteDetails.Reason != "max_output_tokens" {
				s.sawTerminal = true
				s.done = true
				reason := response.IncompleteDetails.Reason
				if reason == "" {
					reason = "unknown"
				}
				return chunk, false, fmt.Errorf("provider/openairesponses: response incomplete: %s", reason)
			}
			chunk.StopReason = agentcore.StopReasonLength
		}
		usage := agentcore.Usage{InputTokens: max(0, response.Usage.InputTokens-response.Usage.Details.Cached), OutputTokens: response.Usage.OutputTokens, CacheReadTokens: response.Usage.Details.Cached}
		chunk.Usage = &usage
		providerData, _ := json.Marshal(output)
		chunk.ProviderData = &agentcore.ProviderData{Format: ProviderDataFormat, Data: providerData}
		s.sawTerminal = true
		s.done = true
		return chunk, false, nil
	case "response.failed", "error":
		s.sawTerminal = true
		s.done = true
		message := base.Error.Message
		if message == "" {
			message = "response failed"
		}
		return chunk, false, fmt.Errorf("provider/openairesponses: %s", message)
	case "response.created", "response.in_progress", "response.content_part.added", "response.content_part.done", "response.function_call_arguments.done":
		return chunk, true, nil
	default:
		return chunk, true, nil
	}
}
func (s *responseStream) indexFor(id string, fallback int) int {
	if id != "" {
		if index, ok := s.callIndex[id]; ok {
			return index
		}
	}
	index := fallback
	if index < 0 {
		index = s.nextIndex
	}
	if index >= s.nextIndex {
		s.nextIndex = index + 1
	}
	if id != "" {
		s.callIndex[id] = index
	}
	return index
}
func (s *responseStream) orderedItems() []any {
	indexes := make([]int, 0, len(s.items))
	for i := range s.items {
		indexes = append(indexes, i)
	}
	sort.Ints(indexes)
	items := make([]any, 0, len(indexes))
	for _, i := range indexes {
		items = append(items, s.items[i])
	}
	return items
}
func (s *responseStream) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.body.Close()
}

var _ agentcore.Model = (*Model)(nil)
var _ agentcore.ModelStream = (*responseStream)(nil)
