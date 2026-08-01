package einoadapter

import (
	"context"
	"encoding/base64"
	"testing"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	"github.com/z3r2ne/agentcore"
)

type fakeEinoTool struct{}

func (fakeEinoTool) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "echo",
		Desc: "echo input",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"text": {Type: schema.String, Required: true},
		}),
	}, nil
}

type fakeEnhancedEinoTool struct{}

func (fakeEnhancedEinoTool) Info(context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{Name: "media"}, nil
}

func (fakeEnhancedEinoTool) InvokableRun(_ context.Context, _ *schema.ToolArgument, _ ...tool.Option) (*schema.ToolResult, error) {
	data := base64.StdEncoding.EncodeToString([]byte("image"))
	return &schema.ToolResult{Parts: []schema.ToolOutputPart{
		{Type: schema.ToolPartTypeText, Text: "result"},
		{Type: schema.ToolPartTypeImage, Image: &schema.ToolOutputImage{MessagePartCommon: schema.MessagePartCommon{Base64Data: &data, MIMEType: "image/png"}}},
	}}, nil
}

func (fakeEinoTool) InvokableRun(_ context.Context, arguments string, _ ...tool.Option) (string, error) {
	return arguments, nil
}

func TestNewToolAdaptsMetadataAndExecution(t *testing.T) {
	adapted, err := NewTool(context.Background(), fakeEinoTool{})
	if err != nil {
		t.Fatal(err)
	}
	definition := adapted.Definition()
	if definition.Name != "echo" || definition.Description != "echo input" || len(definition.Parameters) == 0 {
		t.Fatalf("definition = %+v", definition)
	}
	result, err := adapted.Execute(context.Background(), []byte(`{"text":"hi"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text() != `{"text":"hi"}` {
		t.Fatalf("result = %+v", result)
	}
}

func TestEnhancedToolPreservesMultimodalContent(t *testing.T) {
	adapted, err := NewTool(context.Background(), fakeEnhancedEinoTool{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := adapted.Execute(context.Background(), []byte(`{}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 2 || result.Content[0].Type != agentcore.ContentText || result.Content[1].Type != agentcore.ContentImage || string(result.Content[1].Data) != "image" {
		t.Fatalf("result = %+v", result)
	}
}
