package openai

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
	"strconv"
	"strings"
	"time"

	"github.com/z3r2ne/agentcore"
)

const (
	defaultBaseURL          = "https://api.openai.com/v1"
	defaultMaxResponseBytes = int64(64 << 20)
	defaultMaxErrorBytes    = int64(1 << 20)
	defaultMaxSSEEventBytes = 4 << 20
)

// ProviderDataFormat is the serialized ProviderData format emitted by Model.
const ProviderDataFormat = "openai.chat.completion.message/v1"

// Config controls one OpenAI-compatible Chat Completions model.
//
// Header accepts repeated values and is applied after Headers. Both are copied
// by New, so callers may mutate their input after construction. A custom
// Authorization header overrides the bearer token derived from APIKey.
type Config struct {
	Model      string
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	Headers    map[string]string
	Header     http.Header

	// Limits are applied while reading untrusted provider responses. Zero uses
	// a safe default; negative values are rejected.
	MaxResponseBodyBytes int64
	MaxErrorBodyBytes    int64
	MaxSSEEventBytes     int
}

// Model implements agentcore.Model using streaming Chat Completions.
type Model struct {
	endpoint             string
	apiKey               string
	client               *http.Client
	header               http.Header
	model                string
	maxResponseBodyBytes int64
	maxErrorBodyBytes    int64
	maxSSEEventBytes     int
}

// String deliberately omits credentials and custom headers.
func (m *Model) String() string {
	if m == nil {
		return "openai.Model<nil>"
	}
	return fmt.Sprintf("openai.Model{model:%q, endpoint:%q}", m.model, m.endpoint)
}

// GoString deliberately omits credentials and custom headers.
func (m *Model) GoString() string { return m.String() }

// New constructs a model from Config.
func New(config Config) (*Model, error) {
	endpoint, err := chatCompletionsEndpoint(config.BaseURL)
	if err != nil {
		return nil, err
	}
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, errors.New("provider/openai: model is required")
	}
	if config.MaxResponseBodyBytes < 0 || config.MaxErrorBodyBytes < 0 || config.MaxSSEEventBytes < 0 {
		return nil, errors.New("provider/openai: response limits must not be negative")
	}
	if config.MaxResponseBodyBytes == 0 {
		config.MaxResponseBodyBytes = defaultMaxResponseBytes
	}
	if config.MaxErrorBodyBytes == 0 {
		config.MaxErrorBodyBytes = defaultMaxErrorBytes
	}
	if config.MaxSSEEventBytes == 0 {
		config.MaxSSEEventBytes = defaultMaxSSEEventBytes
	}
	client := config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	header := make(http.Header, len(config.Headers)+len(config.Header))
	for key, value := range config.Headers {
		header.Set(key, value)
	}
	for key, values := range config.Header {
		header.Del(key)
		for _, value := range values {
			header.Add(key, value)
		}
	}
	return &Model{
		endpoint: endpoint, apiKey: strings.TrimSpace(config.APIKey), client: client,
		header: header, model: model, maxResponseBodyBytes: config.MaxResponseBodyBytes,
		maxErrorBodyBytes: config.MaxErrorBodyBytes, maxSSEEventBytes: config.MaxSSEEventBytes,
	}, nil
}

// NewModel is a compatibility-friendly constructor for applications that keep
// the model name separate from provider transport configuration.
func NewModel(config Config, model string) (*Model, error) {
	config.Model = model
	return New(config)
}

// Stream starts one streaming Chat Completions request.
func (m *Model) Stream(ctx context.Context, request agentcore.ModelRequest) (agentcore.ModelStream, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	body, err := m.requestBody(request)
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("provider/openai: encode request: %w", err)
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, m.endpoint, bytes.NewReader(encoded))
	if err != nil {
		return nil, fmt.Errorf("provider/openai: create request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "text/event-stream")
	if m.apiKey != "" {
		httpRequest.Header.Set("Authorization", "Bearer "+m.apiKey)
	}
	for key, values := range m.header {
		httpRequest.Header.Del(key)
		for _, value := range values {
			httpRequest.Header.Add(key, value)
		}
	}

	response, err := m.client.Do(httpRequest)
	if err != nil {
		return nil, &Error{Operation: "request", Retryable: ctx.Err() == nil, Err: err}
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		defer response.Body.Close()
		return nil, readHTTPError(response, m.maxErrorBodyBytes)
	}
	limited := newLimitReadCloser(response.Body, m.maxResponseBodyBytes)
	return newStream(ctx, limited, m.maxSSEEventBytes), nil
}

func (m *Model) requestBody(request agentcore.ModelRequest) (map[string]any, error) {
	messages := make([]map[string]any, 0, len(request.Messages)+1)
	if strings.TrimSpace(request.SystemPrompt) != "" {
		messages = append(messages, map[string]any{"role": "system", "content": request.SystemPrompt})
	}
	for _, message := range request.Messages {
		converted, err := convertMessage(message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	body := map[string]any{
		"model": m.model, "messages": messages, "stream": true,
		"stream_options": map[string]any{"include_usage": true},
	}
	if len(request.Tools) > 0 {
		tools := make([]map[string]any, len(request.Tools))
		for index, definition := range request.Tools {
			parameters := any(map[string]any{"type": "object"})
			if len(definition.Parameters) > 0 {
				decoder := json.NewDecoder(bytes.NewReader(definition.Parameters))
				decoder.UseNumber()
				if err := decoder.Decode(&parameters); err != nil {
					return nil, fmt.Errorf("provider/openai: invalid schema for tool %q: %w", definition.Name, err)
				}
			}
			tools[index] = map[string]any{"type": "function", "function": map[string]any{
				"name": definition.Name, "description": definition.Description, "parameters": parameters,
			}}
		}
		body["tools"] = tools
	}
	for key, value := range request.Options {
		switch key {
		case "model", "messages", "tools", "stream", "stream_options", "n":
			continue
		default:
			body[key] = value
		}
	}
	return body, nil
}

func convertMessage(message agentcore.Message) (map[string]any, error) {
	if message.Role == agentcore.RoleAssistant && message.ProviderData != nil && message.ProviderData.Format == ProviderDataFormat {
		var preserved preservedProviderData
		if err := json.Unmarshal(message.ProviderData.Data, &preserved); err == nil && preserved.Message != nil {
			result := cloneMap(preserved.Message)
			result["role"] = "assistant"
			return result, nil
		}
	}

	result := map[string]any{"role": string(message.Role)}
	switch message.Role {
	case agentcore.RoleTool:
		result["content"] = message.Text()
		result["tool_call_id"] = message.ToolCallID
		if message.ToolName != "" {
			result["name"] = message.ToolName
		}
		return result, nil
	case agentcore.RoleAssistant:
		result["content"] = message.Text()
		if thinking := messageContent(message.Content, agentcore.ContentThinking); thinking != "" {
			result["reasoning_content"] = thinking
		}
		calls := message.ToolCalls()
		if len(calls) > 0 {
			converted := make([]map[string]any, len(calls))
			for index, call := range calls {
				converted[index] = map[string]any{"id": call.ID, "type": "function", "function": map[string]any{
					"name": call.Name, "arguments": string(call.Arguments),
				}}
			}
			result["tool_calls"] = converted
		}
		return result, nil
	case agentcore.RoleSystem, agentcore.RoleUser:
		content, err := convertContent(message.Content)
		if err != nil {
			return nil, err
		}
		result["content"] = content
		return result, nil
	default:
		return nil, fmt.Errorf("provider/openai: unsupported message role %q", message.Role)
	}
}

func messageContent(blocks []agentcore.ContentBlock, kind agentcore.ContentType) string {
	var result string
	for _, block := range blocks {
		if block.Type == kind {
			result += block.Text
		}
	}
	return result
}

func convertContent(blocks []agentcore.ContentBlock) (any, error) {
	if len(blocks) == 0 {
		return "", nil
	}
	if len(blocks) == 1 && blocks[0].Type == agentcore.ContentText {
		return blocks[0].Text, nil
	}
	result := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case agentcore.ContentText:
			result = append(result, map[string]any{"type": "text", "text": block.Text})
		case agentcore.ContentImage:
			imageURL := block.URL
			if imageURL == "" && len(block.Data) > 0 && block.MIMEType != "" {
				imageURL = "data:" + block.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(block.Data)
			}
			if imageURL == "" {
				return nil, errors.New("provider/openai: image URL or MIME-typed data is required")
			}
			result = append(result, map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageURL}})
		default:
			return nil, fmt.Errorf("provider/openai: unsupported user content type %q", block.Type)
		}
	}
	return result, nil
}

func chatCompletionsEndpoint(base string) (string, error) {
	base = strings.TrimSpace(base)
	if base == "" {
		base = defaultBaseURL
	}
	parsed, err := url.Parse(base)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "", errors.New("provider/openai: valid HTTP(S) base URL is required")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	if !strings.HasSuffix(parsed.Path, "/chat/completions") {
		parsed.Path += "/chat/completions"
	}
	return parsed.String(), nil
}

func readHTTPError(response *http.Response, limit int64) error {
	payload, readErr := io.ReadAll(io.LimitReader(response.Body, limit+1))
	truncated := int64(len(payload)) > limit
	if truncated {
		payload = payload[:limit]
	}
	body := strings.TrimSpace(string(payload))
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    any    `json:"code"`
		} `json:"error"`
	}
	providerError := &Error{Operation: "response", StatusCode: response.StatusCode, Body: body, Retryable: retryableStatus(response.StatusCode)}
	if json.Unmarshal(payload, &envelope) == nil && envelope.Error.Message != "" {
		providerError.Body = envelope.Error.Message
		providerError.Type = envelope.Error.Type
		if envelope.Error.Code != nil {
			providerError.Code = fmt.Sprint(envelope.Error.Code)
		}
	}
	if truncated {
		providerError.Body += " …(truncated)"
	}
	if readErr != nil {
		providerError.Err = readErr
	}
	providerError.RetryAfter = parseRetryAfter(response.Header.Get("Retry-After"), time.Now())
	return providerError
}

func parseRetryAfter(value string, now time.Time) time.Duration {
	value = strings.TrimSpace(value)
	if seconds, err := strconv.Atoi(value); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second
	}
	if date, err := http.ParseTime(value); err == nil && date.After(now) {
		return date.Sub(now)
	}
	return 0
}

func cloneMap(input map[string]any) map[string]any {
	result := make(map[string]any, len(input))
	for key, value := range input {
		result[key] = cloneJSONValue(value)
	}
	return result
}

func cloneJSONValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		result := make([]any, len(typed))
		for index := range typed {
			result[index] = cloneJSONValue(typed[index])
		}
		return result
	default:
		return value
	}
}

var _ agentcore.Model = (*Model)(nil)
