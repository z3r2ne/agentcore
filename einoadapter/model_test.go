package einoadapter

import (
	"context"
	"encoding/json"
	"reflect"
	"sync"
	"testing"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/z3r2ne/agentcore"
)

type fakeEinoModel struct {
	mu        sync.Mutex
	responses [][]*schema.Message
	requests  [][]*schema.Message
	tools     []*schema.ToolInfo
	options   []*model.Options
}

func (m *fakeEinoModel) Generate(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error) {
	panic("Generate should not be called")
}

func (m *fakeEinoModel) Stream(_ context.Context, messages []*schema.Message, options ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, messages)
	m.options = append(m.options, model.GetCommonOptions(nil, options...))
	response := m.responses[0]
	m.responses = m.responses[1:]
	return schema.StreamReaderFromArray(response), nil
}

func TestModelResolvesRunLocalOptionsOnEveryCall(t *testing.T) {
	einoModel := &fakeEinoModel{responses: [][]*schema.Message{{{Role: schema.Assistant, Content: "done"}}}}
	var resolved map[string]any
	adapter := Model{
		ChatModel: einoModel,
		Options: func(_ context.Context, options map[string]any) ([]model.Option, error) {
			resolved = options
			return []model.Option{model.WithModel(options["model"].(string)), model.WithMaxTokens(options["max_tokens"].(int))}, nil
		},
	}
	agent, err := agentcore.New(agentcore.Config{Model: adapter, ModelOptions: map[string]any{"model": "dynamic", "max_tokens": 321}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Prompt(context.Background(), agentcore.State{}, []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "go")}, nil); err != nil {
		t.Fatal(err)
	}
	if resolved["model"] != "dynamic" || len(einoModel.options) != 1 || einoModel.options[0].Model == nil || *einoModel.options[0].Model != "dynamic" || einoModel.options[0].MaxTokens == nil || *einoModel.options[0].MaxTokens != 321 {
		t.Fatalf("resolved=%v applied=%+v", resolved, einoModel.options)
	}
}

func (m *fakeEinoModel) WithTools(tools []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	m.tools = tools
	return m, nil
}

func TestModelPreservesEinoProviderDataAcrossToolTurns(t *testing.T) {
	toolIndex := 0
	einoModel := &fakeEinoModel{responses: [][]*schema.Message{
		{
			{
				Role:             schema.Assistant,
				ReasoningContent: "think",
				ToolCalls: []schema.ToolCall{{
					Index: &toolIndex,
					ID:    "call-1",
					Type:  "function",
					Function: schema.FunctionCall{
						Name:      "echo",
						Arguments: `{"text":"hi"}`,
					},
				}},
				Extra: map[string]any{"provider_signature": "preserve-me"},
			},
			{Role: schema.Assistant, ResponseMeta: &schema.ResponseMeta{
				FinishReason: "tool_calls",
				Usage:        &schema.TokenUsage{PromptTokens: 4, CompletionTokens: 2},
			}},
		},
		{{Role: schema.Assistant, Content: "done", ResponseMeta: &schema.ResponseMeta{FinishReason: "stop"}}},
	}}
	echo := agentcore.FuncTool{
		ToolDefinition: agentcore.ToolDefinition{
			Name:        "echo",
			Description: "echo text",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
		},
		ExecuteFunc: func(_ context.Context, arguments json.RawMessage, _ agentcore.ToolUpdateSink) (agentcore.ToolResult, error) {
			return agentcore.TextToolResult(string(arguments)), nil
		},
	}
	agent, err := agentcore.New(agentcore.Config{
		Model:        Model{ChatModel: einoModel},
		SystemPrompt: "system",
		Tools:        []agentcore.Tool{echo},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Prompt(context.Background(), agentcore.State{}, []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "go")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.State.Messages[len(result.State.Messages)-1].Text() != "done" {
		t.Fatalf("result = %+v", result)
	}
	if len(einoModel.tools) != 1 || einoModel.tools[0].Name != "echo" {
		t.Fatalf("tools = %+v", einoModel.tools)
	}
	if len(einoModel.requests) != 2 {
		t.Fatalf("requests = %d", len(einoModel.requests))
	}
	second := einoModel.requests[1]
	if len(second) != 4 {
		t.Fatalf("second request messages = %+v", second)
	}
	if second[0].Role != schema.System || second[0].Content != "system" {
		t.Fatalf("system message = %+v", second[0])
	}
	assistant := second[2]
	if assistant.ReasoningContent != "think" || assistant.Extra["provider_signature"] != "preserve-me" {
		t.Fatalf("provider data was not preserved: %+v", assistant)
	}
	if got := result.State.Messages[1].Usage; !reflect.DeepEqual(got, agentcore.Usage{InputTokens: 4, OutputTokens: 2}) {
		t.Fatalf("usage = %+v", got)
	}
	serialized, err := json.Marshal(result.State)
	if err != nil {
		t.Fatal(err)
	}
	var restored agentcore.State
	if err := json.Unmarshal(serialized, &restored); err != nil {
		t.Fatal(err)
	}
	restoredAssistant, err := toEinoMessage(restored.Messages[1])
	if err != nil {
		t.Fatal(err)
	}
	if restoredAssistant.Extra["provider_signature"] != "preserve-me" {
		t.Fatalf("serialized provider data was not restored: %+v", restoredAssistant)
	}
}

func TestModelConvertsMultimodalInputAndOutput(t *testing.T) {
	imageURL := "https://example.test/result.png"
	einoModel := &fakeEinoModel{
		responses: [][]*schema.Message{
			{
				{
					Role: schema.Assistant,
					AssistantGenMultiContent: []schema.MessageOutputPart{{
						Type: schema.ChatMessagePartTypeText, Text: "image:",
					}, {
						Type:  schema.ChatMessagePartTypeImageURL,
						Image: &schema.MessageOutputImage{MessagePartCommon: schema.MessagePartCommon{URL: &imageURL, MIMEType: "image/png"}},
					}},
					ResponseMeta: &schema.ResponseMeta{FinishReason: "stop"},
				},
			},
		},
	}
	agent, err := agentcore.New(agentcore.Config{Model: Model{ChatModel: einoModel}})
	if err != nil {
		t.Fatal(err)
	}
	prompt := agentcore.Message{Role: agentcore.RoleUser, Content: []agentcore.ContentBlock{
		{Type: agentcore.ContentText, Text: "inspect"},
		{Type: agentcore.ContentImage, Data: []byte("image-bytes"), MIMEType: "image/png"},
	}}
	result, err := agent.Prompt(context.Background(), agentcore.State{}, []agentcore.Message{prompt}, nil)
	if err != nil {
		t.Fatal(err)
	}
	request := einoModel.requests[0][0]
	if len(request.UserInputMultiContent) != 2 || request.UserInputMultiContent[1].Image == nil || request.UserInputMultiContent[1].Image.Base64Data == nil {
		t.Fatalf("multimodal request = %+v", request)
	}
	output := result.State.Messages[1]
	if len(output.Content) != 2 || output.Content[1].Type != agentcore.ContentImage || output.Content[1].URL != imageURL {
		t.Fatalf("multimodal output = %+v", output)
	}
}
