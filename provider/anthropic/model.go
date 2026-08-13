package anthropic

import (
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

	"github.com/z3r2ne/agentcore"
)

const (
	defaultBaseURL     = "https://api.anthropic.com"
	defaultVersion     = "2023-06-01"
	ProviderDataFormat = "anthropic.messages.content/v1"
)

// Config controls an Anthropic Messages model.
type Config struct {
	Model                string
	APIKey               string
	BaseURL              string
	Version              string
	Beta                 []string
	Headers              http.Header
	HTTPClient           *http.Client
	MaxResponseBodyBytes int64
	MaxErrorBodyBytes    int64
	MaxSSEEventBytes     int
}

// Model implements agentcore.Model with Anthropic's native Messages API.
type Model struct {
	model, endpoint, apiKey, version        string
	beta                                    []string
	headers                                 http.Header
	client                                  *http.Client
	maxResponseBodyBytes, maxErrorBodyBytes int64
	maxSSEEventBytes                        int
}

func New(config Config) (*Model, error) {
	if strings.TrimSpace(config.Model) == "" {
		return nil, errors.New("provider/anthropic: model is required")
	}
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = defaultBaseURL
	}
	parsed, err := url.Parse(base)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, errors.New("provider/anthropic: valid HTTP(S) base URL is required")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	if !strings.HasSuffix(parsed.Path, "/v1/messages") {
		parsed.Path += "/v1/messages"
	}
	if config.Version == "" {
		config.Version = defaultVersion
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
		return nil, errors.New("provider/anthropic: response limits must not be negative")
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Model{model: strings.TrimSpace(config.Model), endpoint: parsed.String(), apiKey: strings.TrimSpace(config.APIKey), version: config.Version, beta: append([]string(nil), config.Beta...), headers: config.Headers.Clone(), client: client, maxResponseBodyBytes: config.MaxResponseBodyBytes, maxErrorBodyBytes: config.MaxErrorBodyBytes, maxSSEEventBytes: config.MaxSSEEventBytes}, nil
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
	req.Header.Set("anthropic-version", m.version)
	if m.apiKey != "" {
		req.Header.Set("x-api-key", m.apiKey)
	}
	if len(m.beta) > 0 {
		req.Header.Set("anthropic-beta", strings.Join(m.beta, ","))
	}
	for key, values := range m.headers {
		req.Header.Del(key)
		for _, value := range values {
			req.Header.Add(key, value)
		}
	}
	response, err := m.client.Do(req)
	if err != nil {
		return nil, &Error{Operation: "request", Retryable: ctx.Err() == nil, Err: err}
	}
	if response.StatusCode != http.StatusOK {
		defer response.Body.Close()
		payload, _ := io.ReadAll(io.LimitReader(response.Body, m.maxErrorBodyBytes))
		return nil, decodeHTTPError(response.StatusCode, payload)
	}
	if !strings.HasPrefix(strings.ToLower(response.Header.Get("content-type")), "text/event-stream") {
		defer response.Body.Close()
		return nil, &Error{Operation: "response", StatusCode: response.StatusCode, Message: "expected text/event-stream", Retryable: false}
	}
	return newStream(ctx, &limitedReadCloser{body: response.Body, remaining: m.maxResponseBodyBytes}, m.maxSSEEventBytes), nil
}

func (m *Model) requestBody(request agentcore.ModelRequest) (map[string]any, error) {
	messages := make([]map[string]any, 0, len(request.Messages))
	for _, message := range request.Messages {
		converted, err := convertMessage(message)
		if err != nil {
			return nil, err
		}
		for _, item := range converted {
			messages = appendAnthropicMessage(messages, item)
		}
	}
	body := map[string]any{"model": m.model, "messages": messages, "stream": true, "max_tokens": 4096}
	if request.SystemPrompt != "" {
		body["system"] = request.SystemPrompt
	}
	if len(request.Tools) > 0 {
		tools := make([]map[string]any, len(request.Tools))
		for i, definition := range request.Tools {
			var schema any = map[string]any{"type": "object", "properties": map[string]any{}}
			if len(definition.Parameters) > 0 {
				if err := json.Unmarshal(definition.Parameters, &schema); err != nil {
					return nil, fmt.Errorf("provider/anthropic: invalid schema for tool %q: %w", definition.Name, err)
				}
			}
			tools[i] = map[string]any{"name": definition.Name, "description": definition.Description, "input_schema": schema}
		}
		body["tools"] = tools
	}
	for key, value := range request.Options {
		switch key {
		case "model", "messages", "system", "tools", "stream":
		default:
			body[key] = value
		}
	}
	return body, nil
}

func appendAnthropicMessage(messages []map[string]any, next map[string]any) []map[string]any {
	if len(messages) == 0 || messages[len(messages)-1]["role"] != next["role"] {
		return append(messages, next)
	}
	previous := messages[len(messages)-1]
	previous["content"] = append(anthropicContentBlocks(previous["content"]), anthropicContentBlocks(next["content"])...)
	return messages
}

func anthropicContentBlocks(content any) []any {
	if blocks, ok := content.([]any); ok {
		return blocks
	}
	if text, ok := content.(string); ok {
		return []any{map[string]any{"type": "text", "text": text}}
	}
	return nil
}

func convertMessage(message agentcore.Message) ([]map[string]any, error) {
	if message.Role == agentcore.RoleAssistant && message.ProviderData != nil && message.ProviderData.Format == ProviderDataFormat && json.Valid(message.ProviderData.Data) {
		var content []any
		if json.Unmarshal(message.ProviderData.Data, &content) == nil {
			return []map[string]any{{"role": "assistant", "content": content}}, nil
		}
	}
	switch message.Role {
	case agentcore.RoleSystem:
		return []map[string]any{{"role": "user", "content": message.Text()}}, nil
	case agentcore.RoleUser:
		content, err := convertUserContent(message.Content)
		if err != nil {
			return nil, err
		}
		return []map[string]any{{"role": "user", "content": content}}, nil
	case agentcore.RoleAssistant:
		blocks := make([]any, 0, len(message.Content))
		for _, block := range message.Content {
			switch block.Type {
			case agentcore.ContentText:
				blocks = append(blocks, map[string]any{"type": "text", "text": block.Text})
			case agentcore.ContentThinking:
				if block.Text != "" {
					blocks = append(blocks, map[string]any{"type": "thinking", "thinking": block.Text})
				}
			case agentcore.ContentToolCall:
				if block.ToolCall != nil {
					var input any
					if err := json.Unmarshal(block.ToolCall.Arguments, &input); err != nil {
						return nil, err
					}
					blocks = append(blocks, map[string]any{"type": "tool_use", "id": block.ToolCall.ID, "name": block.ToolCall.Name, "input": input})
				}
			}
		}
		return []map[string]any{{"role": "assistant", "content": blocks}}, nil
	case agentcore.RoleTool:
		content := []any{map[string]any{"type": "tool_result", "tool_use_id": message.ToolCallID, "content": message.Text(), "is_error": message.IsError}}
		return []map[string]any{{"role": "user", "content": content}}, nil
	default:
		return nil, fmt.Errorf("provider/anthropic: unsupported role %q", message.Role)
	}
}

func convertUserContent(content []agentcore.ContentBlock) (any, error) {
	if len(content) == 1 && content[0].Type == agentcore.ContentText {
		return content[0].Text, nil
	}
	blocks := make([]any, 0, len(content))
	for _, block := range content {
		switch block.Type {
		case agentcore.ContentText:
			blocks = append(blocks, map[string]any{"type": "text", "text": block.Text})
		case agentcore.ContentImage:
			if len(block.Data) > 0 && block.MIMEType != "" {
				blocks = append(blocks, map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": block.MIMEType, "data": base64.StdEncoding.EncodeToString(block.Data)}})
			} else {
				return nil, errors.New("provider/anthropic: images require MIME-typed inline data")
			}
		default:
			return nil, fmt.Errorf("provider/anthropic: unsupported user content type %q", block.Type)
		}
	}
	return blocks, nil
}

func decodeHTTPError(status int, payload []byte) error {
	var envelope struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal(payload, &envelope)
	message := envelope.Error.Message
	if message == "" {
		message = strings.TrimSpace(string(payload))
	}
	return &Error{Operation: "response", StatusCode: status, Type: envelope.Error.Type, Message: message, Retryable: retryableStatus(status)}
}

var _ agentcore.Model = (*Model)(nil)
