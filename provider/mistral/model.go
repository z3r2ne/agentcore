package mistral

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/z3r2ne/agentcore"
	chatprovider "github.com/z3r2ne/agentcore/provider/openai"
)

type Config struct {
	Model, APIKey, BaseURL string
	Headers                http.Header
	HTTPClient             *http.Client
}
type Model struct{ base agentcore.Model }

func New(config Config) (*Model, error) {
	base := config.BaseURL
	if strings.TrimSpace(base) == "" {
		base = "https://api.mistral.ai/v1"
	}
	model, err := chatprovider.New(chatprovider.Config{
		Model: config.Model, APIKey: config.APIKey, BaseURL: base,
		Header: config.Headers, HTTPClient: config.HTTPClient,
		BuildRequestBody: mistralRequestBody,
	})
	if err != nil {
		return nil, err
	}
	return &Model{base: model}, nil
}
func (m *Model) Stream(ctx context.Context, request agentcore.ModelRequest) (agentcore.ModelStream, error) {
	request.Messages = normalizeMessages(request.Messages)
	return m.base.Stream(ctx, request)
}
func normalizeMessages(messages []agentcore.Message) []agentcore.Message {
	result := make([]agentcore.Message, len(messages))
	for i, message := range messages {
		message.Content = append([]agentcore.ContentBlock(nil), message.Content...)
		changed := false
		for j, block := range message.Content {
			if block.ToolCall != nil {
				call := *block.ToolCall
				normalized := normalizeID(call.ID)
				if normalized != call.ID {
					changed = true
					call.ID = normalized
				}
				call.Arguments = append([]byte(nil), call.Arguments...)
				block.ToolCall = &call
				message.Content[j] = block
			}
		}
		if message.ToolCallID != "" {
			normalized := normalizeID(message.ToolCallID)
			if normalized != message.ToolCallID {
				changed = true
				message.ToolCallID = normalized
			}
		}
		if changed {
			message.ProviderData = nil
		}
		result[i] = message
	}
	return result
}

var validID = regexp.MustCompile(`^[A-Za-z0-9]{9}$`)

func normalizeID(id string) string {
	if validID.MatchString(id) {
		return id
	}
	sum := sha256.Sum256([]byte(id))
	return fmt.Sprintf("%x", sum[:])[:9]
}

func mistralRequestBody(model string, request agentcore.ModelRequest) (map[string]any, error) {
	messages := make([]any, 0, len(request.Messages)+1)
	if request.SystemPrompt != "" {
		messages = append(messages, map[string]any{"role": "system", "content": request.SystemPrompt})
	}
	for _, message := range request.Messages {
		converted, err := mistralMessage(message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	body := map[string]any{"model": model, "messages": messages, "stream": true}
	if len(request.Tools) > 0 {
		tools := make([]any, len(request.Tools))
		for index, tool := range request.Tools {
			var parameters any = map[string]any{"type": "object", "properties": map[string]any{}}
			if len(tool.Parameters) > 0 {
				if err := json.Unmarshal(tool.Parameters, &parameters); err != nil {
					return nil, fmt.Errorf("provider/mistral: invalid schema for tool %q: %w", tool.Name, err)
				}
			}
			tools[index] = map[string]any{"type": "function", "function": map[string]any{
				"name": tool.Name, "description": tool.Description, "parameters": parameters, "strict": true,
			}}
		}
		body["tools"] = tools
	}
	for key, value := range request.Options {
		if key == "model" || key == "messages" || key == "tools" || key == "stream" || key == "stream_options" || key == "n" {
			continue
		}
		body[mistralOptionName(key)] = mistralOptionValue(key, value)
	}
	return body, nil
}

func mistralMessage(message agentcore.Message) (map[string]any, error) {
	result := map[string]any{"role": string(message.Role)}
	switch message.Role {
	case agentcore.RoleSystem:
		result["content"] = message.Text()
	case agentcore.RoleUser:
		content, err := mistralContent(message.Content)
		if err != nil {
			return nil, err
		}
		result["content"] = content
	case agentcore.RoleAssistant:
		content := make([]any, 0, len(message.Content))
		calls := make([]any, 0)
		for _, block := range message.Content {
			switch block.Type {
			case agentcore.ContentThinking:
				content = append(content, map[string]any{"type": "thinking", "thinking": []any{map[string]any{"type": "text", "text": block.Text}}})
			case agentcore.ContentText:
				content = append(content, map[string]any{"type": "text", "text": block.Text})
			case agentcore.ContentToolCall:
				if block.ToolCall == nil {
					continue
				}
				calls = append(calls, map[string]any{"id": block.ToolCall.ID, "type": "function", "index": len(calls), "function": map[string]any{
					"name": block.ToolCall.Name, "arguments": string(block.ToolCall.Arguments),
				}})
			default:
				return nil, fmt.Errorf("provider/mistral: unsupported assistant content type %q", block.Type)
			}
		}
		result["prefix"] = false
		result["content"] = content
		if len(calls) > 0 {
			result["tool_calls"] = calls
		}
	case agentcore.RoleTool:
		content, err := mistralContent(message.Content)
		if err != nil {
			return nil, err
		}
		result["content"] = content
		result["tool_call_id"] = message.ToolCallID
		result["name"] = message.ToolName
	default:
		return nil, fmt.Errorf("provider/mistral: unsupported role %q", message.Role)
	}
	return result, nil
}

func mistralContent(blocks []agentcore.ContentBlock) (any, error) {
	if len(blocks) == 1 && blocks[0].Type == agentcore.ContentText {
		return blocks[0].Text, nil
	}
	content := make([]any, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case agentcore.ContentText:
			content = append(content, map[string]any{"type": "text", "text": block.Text})
		case agentcore.ContentImage:
			imageURL := block.URL
			if imageURL == "" && len(block.Data) > 0 && block.MIMEType != "" {
				imageURL = "data:" + block.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(block.Data)
			}
			if imageURL == "" {
				return nil, fmt.Errorf("provider/mistral: image requires URL or MIME-typed data")
			}
			content = append(content, map[string]any{"type": "image_url", "image_url": imageURL})
		default:
			return nil, fmt.Errorf("provider/mistral: unsupported content type %q", block.Type)
		}
	}
	return content, nil
}

func mistralOptionName(name string) string {
	aliases := map[string]string{
		"topP": "top_p", "maxTokens": "max_tokens", "randomSeed": "random_seed",
		"responseFormat": "response_format", "toolChoice": "tool_choice",
		"presencePenalty": "presence_penalty", "frequencyPenalty": "frequency_penalty",
		"parallelToolCalls": "parallel_tool_calls", "reasoningEffort": "reasoning_effort",
		"promptMode": "prompt_mode", "promptCacheKey": "prompt_cache_key", "safePrompt": "safe_prompt",
	}
	if alias := aliases[name]; alias != "" {
		return alias
	}
	return name
}

func mistralOptionValue(name string, value any) any {
	if name != "responseFormat" {
		return value
	}
	format, ok := value.(map[string]any)
	if !ok {
		return value
	}
	converted := cloneAnyMap(format)
	if schema, ok := converted["jsonSchema"]; ok {
		delete(converted, "jsonSchema")
		if definition, ok := schema.(map[string]any); ok {
			definition = cloneAnyMap(definition)
			if value, exists := definition["schemaDefinition"]; exists {
				delete(definition, "schemaDefinition")
				definition["schema"] = value
			}
			schema = definition
		}
		converted["json_schema"] = schema
	}
	return converted
}

func cloneAnyMap(input map[string]any) map[string]any {
	result := make(map[string]any, len(input))
	for key, value := range input {
		result[key] = value
	}
	return result
}

var _ agentcore.Model = (*Model)(nil)
