package einoadapter

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	jsonschema "github.com/eino-contrib/jsonschema"
	"github.com/z3r2ne/agentcore"
)

// Model adapts an immutable Eino ToolCallingChatModel to agentcore.Model.
type Model struct {
	ChatModel model.ToolCallingChatModel
	// Options maps run-local settings to Eino options on every model attempt.
	Options func(context.Context, map[string]any) ([]model.Option, error)
}

func (m Model) Stream(ctx context.Context, request agentcore.ModelRequest) (agentcore.ModelStream, error) {
	if m.ChatModel == nil {
		return nil, errors.New("einoadapter: nil Eino chat model")
	}
	infos := make([]*schema.ToolInfo, 0, len(request.Tools))
	for _, definition := range request.Tools {
		info, err := toolInfo(definition)
		if err != nil {
			return nil, err
		}
		infos = append(infos, info)
	}
	chatModel := m.ChatModel
	if len(infos) > 0 {
		var err error
		chatModel, err = chatModel.WithTools(infos)
		if err != nil {
			return nil, fmt.Errorf("einoadapter: bind tools: %w", err)
		}
	}
	messages := make([]*schema.Message, 0, len(request.Messages)+1)
	if request.SystemPrompt != "" {
		messages = append(messages, schema.SystemMessage(request.SystemPrompt))
	}
	for _, message := range request.Messages {
		converted, err := toEinoMessage(message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	var options []model.Option
	if m.Options != nil {
		resolvedOptions, err := m.Options(ctx, request.Options)
		if err != nil {
			return nil, fmt.Errorf("einoadapter: model options: %w", err)
		}
		options = resolvedOptions
	}
	stream, err := chatModel.Stream(ctx, messages, options...)
	if err != nil {
		return nil, err
	}
	return &modelStream{stream: stream}, nil
}

type modelStream struct {
	stream *schema.StreamReader[*schema.Message]
	merged *schema.Message
}

func (s *modelStream) Recv() (agentcore.ModelChunk, error) {
	message, err := s.stream.Recv()
	if err != nil {
		return agentcore.ModelChunk{}, err
	}
	chunk := agentcore.ModelChunk{
		TextDelta:     message.Content,
		ThinkingDelta: message.ReasoningContent,
	}
	if len(message.AssistantGenMultiContent) > 0 {
		chunk.TextDelta = ""
		chunk.ThinkingDelta = ""
		for _, part := range message.AssistantGenMultiContent {
			if block, ok := outputPartToCore(part); ok {
				chunk.ContentDeltas = append(chunk.ContentDeltas, block)
			}
		}
	}
	toMerge := []*schema.Message{message}
	if s.merged != nil {
		toMerge = append([]*schema.Message{s.merged}, toMerge...)
	}
	s.merged, err = schema.ConcatMessages(toMerge)
	if err != nil {
		return agentcore.ModelChunk{}, fmt.Errorf("einoadapter: merge model chunks: %w", err)
	}
	serialized, err := json.Marshal(s.merged)
	if err != nil {
		return agentcore.ModelChunk{}, fmt.Errorf("einoadapter: serialize provider message: %w", err)
	}
	chunk.ProviderData = &agentcore.ProviderData{
		Format:  "eino.schema.Message/v1",
		Data:    serialized,
		Runtime: s.merged,
	}
	for position, call := range message.ToolCalls {
		index := position
		if call.Index != nil {
			index = *call.Index
		}
		chunk.ToolCallDeltas = append(chunk.ToolCallDeltas, agentcore.ToolCallDelta{
			Index:          index,
			ID:             call.ID,
			Name:           call.Function.Name,
			ArgumentsDelta: call.Function.Arguments,
		})
	}
	if message.ResponseMeta != nil {
		chunk.StopReason = normalizeStopReason(message.ResponseMeta.FinishReason)
		if message.ResponseMeta.Usage != nil {
			chunk.Usage = &agentcore.Usage{
				InputTokens:     message.ResponseMeta.Usage.PromptTokens,
				OutputTokens:    message.ResponseMeta.Usage.CompletionTokens,
				CacheReadTokens: message.ResponseMeta.Usage.PromptTokenDetails.CachedTokens,
			}
		}
	}
	return chunk, nil
}

func (s *modelStream) Close() error {
	s.stream.Close()
	return nil
}

func toolInfo(definition agentcore.ToolDefinition) (*schema.ToolInfo, error) {
	info := &schema.ToolInfo{Name: definition.Name, Desc: definition.Description}
	if len(definition.Parameters) == 0 {
		return info, nil
	}
	var parameters jsonschema.Schema
	if err := json.Unmarshal(definition.Parameters, &parameters); err != nil {
		return nil, fmt.Errorf("einoadapter: invalid JSON schema for tool %q: %w", definition.Name, err)
	}
	info.ParamsOneOf = schema.NewParamsOneOfByJSONSchema(&parameters)
	return info, nil
}

func toEinoMessage(message agentcore.Message) (*schema.Message, error) {
	if message.ProviderData != nil && message.ProviderData.Format == "eino.schema.Message/v1" {
		if preserved, ok := message.ProviderData.Runtime.(*schema.Message); ok && preserved != nil {
			return preserved, nil
		}
		if len(message.ProviderData.Data) > 0 {
			var preserved schema.Message
			if err := json.Unmarshal(message.ProviderData.Data, &preserved); err != nil {
				return nil, fmt.Errorf("einoadapter: restore provider message: %w", err)
			}
			return &preserved, nil
		}
	}
	converted := &schema.Message{
		Role:       schema.RoleType(message.Role),
		ToolCallID: message.ToolCallID,
		ToolName:   message.ToolName,
	}
	var inputParts []schema.MessageInputPart
	var outputParts []schema.MessageOutputPart
	hasInputMedia := false
	for _, block := range message.Content {
		switch block.Type {
		case agentcore.ContentText:
			converted.Content += block.Text
			inputParts = append(inputParts, schema.MessageInputPart{Type: schema.ChatMessagePartTypeText, Text: block.Text})
			outputParts = append(outputParts, schema.MessageOutputPart{Type: schema.ChatMessagePartTypeText, Text: block.Text})
		case agentcore.ContentThinking:
			converted.ReasoningContent += block.Text
			outputParts = append(outputParts, schema.MessageOutputPart{
				Type:      schema.ChatMessagePartTypeReasoning,
				Reasoning: &schema.MessageOutputReasoning{Text: block.Text},
			})
		case agentcore.ContentImage, agentcore.ContentAudio, agentcore.ContentVideo, agentcore.ContentFile:
			if message.Role == agentcore.RoleAssistant && block.Type == agentcore.ContentFile {
				return nil, errors.New("einoadapter: assistant file output is not supported by Eino schema.Message")
			}
			part, err := coreToInputPart(block)
			if err != nil {
				return nil, err
			}
			inputParts = append(inputParts, part)
			hasInputMedia = true
			if message.Role == agentcore.RoleAssistant && block.Type != agentcore.ContentFile {
				output, err := coreToOutputPart(block)
				if err != nil {
					return nil, err
				}
				outputParts = append(outputParts, output)
			}
		case agentcore.ContentToolCall:
			if block.ToolCall == nil {
				continue
			}
			index := len(converted.ToolCalls)
			converted.ToolCalls = append(converted.ToolCalls, schema.ToolCall{
				Index: &index,
				ID:    block.ToolCall.ID,
				Type:  "function",
				Function: schema.FunctionCall{
					Name:      block.ToolCall.Name,
					Arguments: string(block.ToolCall.Arguments),
				},
			})
		default:
			return nil, fmt.Errorf("einoadapter: unsupported content type %q", block.Type)
		}
	}
	if hasInputMedia && message.Role != agentcore.RoleAssistant {
		converted.Content = ""
		converted.UserInputMultiContent = inputParts
	}
	if message.Role == agentcore.RoleAssistant && len(outputParts) > 0 && hasCoreMedia(message.Content) {
		converted.Content = ""
		converted.ReasoningContent = ""
		converted.AssistantGenMultiContent = outputParts
	}
	return converted, nil
}

func coreToInputPart(block agentcore.ContentBlock) (schema.MessageInputPart, error) {
	common := schema.MessagePartCommon{MIMEType: block.MIMEType}
	if block.URL != "" {
		url := block.URL
		common.URL = &url
	}
	if len(block.Data) > 0 {
		data := base64.StdEncoding.EncodeToString(block.Data)
		common.Base64Data = &data
	}
	switch block.Type {
	case agentcore.ContentImage:
		return schema.MessageInputPart{Type: schema.ChatMessagePartTypeImageURL, Image: &schema.MessageInputImage{MessagePartCommon: common}}, nil
	case agentcore.ContentAudio:
		return schema.MessageInputPart{Type: schema.ChatMessagePartTypeAudioURL, Audio: &schema.MessageInputAudio{MessagePartCommon: common}}, nil
	case agentcore.ContentVideo:
		return schema.MessageInputPart{Type: schema.ChatMessagePartTypeVideoURL, Video: &schema.MessageInputVideo{MessagePartCommon: common}}, nil
	case agentcore.ContentFile:
		return schema.MessageInputPart{Type: schema.ChatMessagePartTypeFileURL, File: &schema.MessageInputFile{MessagePartCommon: common, Name: block.Name}}, nil
	default:
		return schema.MessageInputPart{}, fmt.Errorf("einoadapter: unsupported input content type %q", block.Type)
	}
}

func coreToOutputPart(block agentcore.ContentBlock) (schema.MessageOutputPart, error) {
	common := schema.MessagePartCommon{MIMEType: block.MIMEType}
	if block.URL != "" {
		url := block.URL
		common.URL = &url
	}
	if len(block.Data) > 0 {
		data := base64.StdEncoding.EncodeToString(block.Data)
		common.Base64Data = &data
	}
	switch block.Type {
	case agentcore.ContentImage:
		return schema.MessageOutputPart{Type: schema.ChatMessagePartTypeImageURL, Image: &schema.MessageOutputImage{MessagePartCommon: common}}, nil
	case agentcore.ContentAudio:
		return schema.MessageOutputPart{Type: schema.ChatMessagePartTypeAudioURL, Audio: &schema.MessageOutputAudio{MessagePartCommon: common}}, nil
	case agentcore.ContentVideo:
		return schema.MessageOutputPart{Type: schema.ChatMessagePartTypeVideoURL, Video: &schema.MessageOutputVideo{MessagePartCommon: common}}, nil
	default:
		return schema.MessageOutputPart{}, fmt.Errorf("einoadapter: unsupported output content type %q", block.Type)
	}
}

func outputPartToCore(part schema.MessageOutputPart) (agentcore.ContentBlock, bool) {
	switch part.Type {
	case schema.ChatMessagePartTypeText:
		return agentcore.ContentBlock{Type: agentcore.ContentText, Text: part.Text}, true
	case schema.ChatMessagePartTypeReasoning:
		if part.Reasoning == nil {
			return agentcore.ContentBlock{}, false
		}
		return agentcore.ContentBlock{Type: agentcore.ContentThinking, Text: part.Reasoning.Text}, true
	case schema.ChatMessagePartTypeImageURL:
		if part.Image == nil {
			return agentcore.ContentBlock{}, false
		}
		return commonToCore(agentcore.ContentImage, part.Image.MessagePartCommon, ""), true
	case schema.ChatMessagePartTypeAudioURL:
		if part.Audio == nil {
			return agentcore.ContentBlock{}, false
		}
		return commonToCore(agentcore.ContentAudio, part.Audio.MessagePartCommon, ""), true
	case schema.ChatMessagePartTypeVideoURL:
		if part.Video == nil {
			return agentcore.ContentBlock{}, false
		}
		return commonToCore(agentcore.ContentVideo, part.Video.MessagePartCommon, ""), true
	default:
		return agentcore.ContentBlock{}, false
	}
}

func commonToCore(kind agentcore.ContentType, common schema.MessagePartCommon, name string) agentcore.ContentBlock {
	block := agentcore.ContentBlock{Type: kind, MIMEType: common.MIMEType, Name: name}
	if common.URL != nil {
		block.URL = *common.URL
	}
	if common.Base64Data != nil {
		if data, err := base64.StdEncoding.DecodeString(*common.Base64Data); err == nil {
			block.Data = data
		}
	}
	return block
}

func hasCoreMedia(content []agentcore.ContentBlock) bool {
	for _, block := range content {
		if block.Type == agentcore.ContentImage || block.Type == agentcore.ContentAudio || block.Type == agentcore.ContentVideo || block.Type == agentcore.ContentFile {
			return true
		}
	}
	return false
}

func normalizeStopReason(reason string) agentcore.StopReason {
	switch strings.ToLower(reason) {
	case "":
		return ""
	case "stop", "end_turn", "stop_sequence":
		return agentcore.StopReasonStop
	case "tool_calls", "tool_use":
		return agentcore.StopReasonToolUse
	case "length", "max_tokens", "max_output_tokens":
		return agentcore.StopReasonLength
	case "error", "content_filter":
		return agentcore.StopReasonError
	default:
		return agentcore.StopReason(reason)
	}
}
