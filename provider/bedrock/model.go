package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/z3r2ne/agentcore"
)

// Config controls an AWS Bedrock ConverseStream model. When Client is nil,
// New loads the standard AWS credential chain; Region and Profile optionally
// override its resolution. BaseEndpoint supports private/VPC test endpoints.
type Config struct {
	Model, Region, Profile, BaseEndpoint string
	Client                               *bedrockruntime.Client
}
type Model struct {
	model  string
	client *bedrockruntime.Client
}

const ProviderDataFormat = "bedrock.converse.content/v1"

type preservedBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	ToolID    string `json:"toolId,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// New creates a Bedrock model and loads AWS configuration when needed.
func New(ctx context.Context, config Config) (*Model, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, errors.New("provider/bedrock: model is required")
	}
	client := config.Client
	if client == nil {
		options := []func(*awsconfig.LoadOptions) error{}
		if config.Region != "" {
			options = append(options, awsconfig.WithRegion(config.Region))
		}
		if config.Profile != "" {
			options = append(options, awsconfig.WithSharedConfigProfile(config.Profile))
		}
		awsCfg, err := awsconfig.LoadDefaultConfig(ctx, options...)
		if err != nil {
			return nil, fmt.Errorf("provider/bedrock: load AWS config: %w", err)
		}
		client = bedrockruntime.NewFromConfig(awsCfg, func(options *bedrockruntime.Options) {
			if config.BaseEndpoint != "" {
				options.BaseEndpoint = &config.BaseEndpoint
			}
		})
	}
	return &Model{model: model, client: client}, nil
}

func (m *Model) Stream(ctx context.Context, request agentcore.ModelRequest) (agentcore.ModelStream, error) {
	input, err := m.input(request)
	if err != nil {
		return nil, err
	}
	output, err := m.client.ConverseStream(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("provider/bedrock: ConverseStream: %w", err)
	}
	if output == nil || output.GetStream() == nil {
		return nil, errors.New("provider/bedrock: nil ConverseStream output")
	}
	return &stream{events: output.GetStream(), calls: map[int]agentcore.ToolCall{}, blocks: map[int]*preservedBlock{}}, nil
}

func (m *Model) input(request agentcore.ModelRequest) (*bedrockruntime.ConverseStreamInput, error) {
	messages := make([]bedrocktypes.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		converted, err := bedrockMessage(message)
		if err != nil {
			return nil, err
		}
		if len(messages) > 0 && messages[len(messages)-1].Role == converted.Role {
			messages[len(messages)-1].Content = append(messages[len(messages)-1].Content, converted.Content...)
		} else {
			messages = append(messages, converted)
		}
	}
	input := &bedrockruntime.ConverseStreamInput{ModelId: aws.String(m.model), Messages: messages}
	if request.SystemPrompt != "" {
		input.System = []bedrocktypes.SystemContentBlock{&bedrocktypes.SystemContentBlockMemberText{Value: request.SystemPrompt}}
	}
	if len(request.Tools) > 0 {
		tools := make([]bedrocktypes.Tool, len(request.Tools))
		for i, definition := range request.Tools {
			var schema any = map[string]any{"type": "object", "properties": map[string]any{}}
			if len(definition.Parameters) > 0 {
				if err := json.Unmarshal(definition.Parameters, &schema); err != nil {
					return nil, err
				}
			}
			tools[i] = &bedrocktypes.ToolMemberToolSpec{Value: bedrocktypes.ToolSpecification{Name: aws.String(definition.Name), Description: aws.String(definition.Description), InputSchema: &bedrocktypes.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(schema)}}}
		}
		input.ToolConfig = &bedrocktypes.ToolConfiguration{Tools: tools}
	}
	inference := &bedrocktypes.InferenceConfiguration{}
	configured := false
	if value, ok := intOption(request.Options, "maxTokens", "max_tokens", "max_output_tokens"); ok {
		tokens := int32(value)
		inference.MaxTokens = &tokens
		configured = true
	}
	if value, ok := floatOption(request.Options, "temperature"); ok {
		v := float32(value)
		inference.Temperature = &v
		configured = true
	}
	if value, ok := floatOption(request.Options, "topP", "top_p"); ok {
		v := float32(value)
		inference.TopP = &v
		configured = true
	}
	if value, ok := request.Options["stopSequences"].([]string); ok {
		inference.StopSequences = append([]string(nil), value...)
		configured = true
	}
	if configured {
		input.InferenceConfig = inference
	}
	return input, nil
}

func bedrockMessage(message agentcore.Message) (bedrocktypes.Message, error) {
	role := bedrocktypes.ConversationRoleUser
	if message.Role == agentcore.RoleAssistant {
		role = bedrocktypes.ConversationRoleAssistant
	}
	content := make([]bedrocktypes.ContentBlock, 0, len(message.Content))
	if message.Role == agentcore.RoleAssistant && message.ProviderData != nil && message.ProviderData.Format == ProviderDataFormat && json.Valid(message.ProviderData.Data) {
		var blocks []preservedBlock
		if json.Unmarshal(message.ProviderData.Data, &blocks) == nil {
			for _, block := range blocks {
				switch block.Type {
				case "text":
					content = append(content, &bedrocktypes.ContentBlockMemberText{Value: block.Text})
				case "thinking":
					content = append(content, &bedrocktypes.ContentBlockMemberReasoningContent{Value: &bedrocktypes.ReasoningContentBlockMemberReasoningText{Value: bedrocktypes.ReasoningTextBlock{Text: aws.String(block.Thinking), Signature: aws.String(block.Signature)}}})
				case "tool_use":
					var input any
					if json.Unmarshal([]byte(block.Arguments), &input) != nil {
						input = map[string]any{}
					}
					content = append(content, &bedrocktypes.ContentBlockMemberToolUse{Value: bedrocktypes.ToolUseBlock{ToolUseId: aws.String(block.ToolID), Name: aws.String(block.Name), Input: document.NewLazyDocument(input)}})
				}
			}
			return bedrocktypes.Message{Role: role, Content: content}, nil
		}
	}
	switch message.Role {
	case agentcore.RoleSystem, agentcore.RoleUser, agentcore.RoleAssistant:
		for _, block := range message.Content {
			switch block.Type {
			case agentcore.ContentText:
				content = append(content, &bedrocktypes.ContentBlockMemberText{Value: block.Text})
			case agentcore.ContentToolCall:
				if block.ToolCall != nil {
					var input any
					if err := json.Unmarshal(block.ToolCall.Arguments, &input); err != nil {
						return bedrocktypes.Message{}, err
					}
					content = append(content, &bedrocktypes.ContentBlockMemberToolUse{Value: bedrocktypes.ToolUseBlock{
						ToolUseId: aws.String(block.ToolCall.ID), Name: aws.String(block.ToolCall.Name), Input: document.NewLazyDocument(input),
					}})
				}
			}
		}
	case agentcore.RoleTool:
		status := bedrocktypes.ToolResultStatusSuccess
		if message.IsError {
			status = bedrocktypes.ToolResultStatusError
		}
		content = append(content, &bedrocktypes.ContentBlockMemberToolResult{Value: bedrocktypes.ToolResultBlock{
			ToolUseId: aws.String(message.ToolCallID), Status: status,
			Content: []bedrocktypes.ToolResultContentBlock{&bedrocktypes.ToolResultContentBlockMemberText{Value: message.Text()}},
		}})
	default:
		return bedrocktypes.Message{}, fmt.Errorf("provider/bedrock: unsupported role %q", message.Role)
	}
	return bedrocktypes.Message{Role: role, Content: content}, nil
}

func intOption(options map[string]any, keys ...string) (int, bool) {
	for _, key := range keys {
		switch value := options[key].(type) {
		case int:
			return value, true
		case float64:
			return int(value), true
		case json.Number:
			n, err := value.Int64()
			return int(n), err == nil
		}
	}
	return 0, false
}
func floatOption(options map[string]any, keys ...string) (float64, bool) {
	for _, key := range keys {
		switch value := options[key].(type) {
		case float64:
			return value, true
		case float32:
			return float64(value), true
		case int:
			return float64(value), true
		case json.Number:
			n, err := value.Float64()
			return n, err == nil
		}
	}
	return 0, false
}

type stream struct {
	events          *bedrockruntime.ConverseStreamEventStream
	calls           map[int]agentcore.ToolCall
	blocks          map[int]*preservedBlock
	order           []int
	sawStop, closed bool
}

func (s *stream) Recv() (agentcore.ModelChunk, error) {
	if s.closed {
		return agentcore.ModelChunk{}, io.EOF
	}
	for event := range s.events.Events() {
		chunk := agentcore.ModelChunk{}
		switch typed := event.(type) {
		case *bedrocktypes.ConverseStreamOutputMemberContentBlockStart:
			index := int(*typed.Value.ContentBlockIndex)
			if tool, ok := typed.Value.Start.(*bedrocktypes.ContentBlockStartMemberToolUse); ok {
				call := agentcore.ToolCall{ID: aws.ToString(tool.Value.ToolUseId), Name: aws.ToString(tool.Value.Name)}
				s.calls[index] = call
				s.ensureBlock(index, &preservedBlock{Type: "tool_use", ToolID: call.ID, Name: call.Name})
				chunk.ToolCallDeltas = []agentcore.ToolCallDelta{{Index: index, ID: call.ID, Name: call.Name}}
				return chunk, nil
			}
		case *bedrocktypes.ConverseStreamOutputMemberContentBlockDelta:
			index := int(*typed.Value.ContentBlockIndex)
			switch delta := typed.Value.Delta.(type) {
			case *bedrocktypes.ContentBlockDeltaMemberText:
				block := s.ensureBlock(index, &preservedBlock{Type: "text"})
				block.Text += delta.Value
				chunk.TextDelta = delta.Value
				return chunk, nil
			case *bedrocktypes.ContentBlockDeltaMemberToolUse:
				block := s.ensureBlock(index, &preservedBlock{Type: "tool_use"})
				block.Arguments += aws.ToString(delta.Value.Input)
				chunk.ToolCallDeltas = []agentcore.ToolCallDelta{{Index: index, ArgumentsDelta: aws.ToString(delta.Value.Input)}}
				return chunk, nil
			case *bedrocktypes.ContentBlockDeltaMemberReasoningContent:
				switch reasoning := delta.Value.(type) {
				case *bedrocktypes.ReasoningContentBlockDeltaMemberText:
					block := s.ensureBlock(index, &preservedBlock{Type: "thinking"})
					block.Thinking += reasoning.Value
					chunk.ThinkingDelta = reasoning.Value
					return chunk, nil
				case *bedrocktypes.ReasoningContentBlockDeltaMemberSignature:
					block := s.ensureBlock(index, &preservedBlock{Type: "thinking"})
					block.Signature += reasoning.Value
				}
			}
		case *bedrocktypes.ConverseStreamOutputMemberMessageStop:
			s.sawStop = true
			chunk.StopReason = bedrockStop(string(typed.Value.StopReason))
			data, err := json.Marshal(s.orderedBlocks())
			if err != nil {
				return agentcore.ModelChunk{}, err
			}
			chunk.ProviderData = &agentcore.ProviderData{Format: ProviderDataFormat, Data: data}
			return chunk, nil
		case *bedrocktypes.ConverseStreamOutputMemberMetadata:
			if typed.Value.Usage != nil {
				usage := typed.Value.Usage
				read := int(aws.ToInt32(usage.CacheReadInputTokens))
				write := int(aws.ToInt32(usage.CacheWriteInputTokens))
				input := int(aws.ToInt32(usage.InputTokens)) - read - write
				if input < 0 {
					input = 0
				}
				normalized := agentcore.Usage{InputTokens: input, OutputTokens: int(aws.ToInt32(usage.OutputTokens)), CacheReadTokens: read, CacheWriteTokens: write}
				chunk.Usage = &normalized
				return chunk, nil
			}
		}
	}
	s.closed = true
	if err := s.events.Err(); err != nil {
		return agentcore.ModelChunk{}, fmt.Errorf("provider/bedrock: stream: %w", err)
	}
	if !s.sawStop {
		return agentcore.ModelChunk{}, io.ErrUnexpectedEOF
	}
	return agentcore.ModelChunk{}, io.EOF
}

func (s *stream) ensureBlock(index int, initial *preservedBlock) *preservedBlock {
	if block := s.blocks[index]; block != nil {
		return block
	}
	s.blocks[index] = initial
	s.order = append(s.order, index)
	return initial
}

func (s *stream) orderedBlocks() []preservedBlock {
	result := make([]preservedBlock, 0, len(s.order))
	for _, index := range s.order {
		if block := s.blocks[index]; block != nil {
			result = append(result, *block)
		}
	}
	return result
}
func bedrockStop(reason string) agentcore.StopReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return agentcore.StopReasonStop
	case "tool_use":
		return agentcore.StopReasonToolUse
	case "max_tokens", "model_context_window_exceeded":
		return agentcore.StopReasonLength
	case "guardrail_intervened", "content_filtered":
		return agentcore.StopReasonError
	default:
		return agentcore.StopReason(reason)
	}
}
func (s *stream) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.events.Close()
}

var _ agentcore.Model = (*Model)(nil)
var _ agentcore.ModelStream = (*stream)(nil)
