package einoadapter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	"github.com/z3r2ne/agentcore"
)

// Tool adapts an existing Eino invokable or streamable tool.
type Tool struct {
	base       tool.BaseTool
	definition agentcore.ToolDefinition
}

// NewTool reads Eino tool metadata and returns an agentcore-compatible tool.
func NewTool(ctx context.Context, base tool.BaseTool) (*Tool, error) {
	if base == nil {
		return nil, errors.New("einoadapter: nil tool")
	}
	info, err := base.Info(ctx)
	if err != nil {
		return nil, fmt.Errorf("einoadapter: read tool info: %w", err)
	}
	definition := agentcore.ToolDefinition{Name: info.Name, Description: info.Desc}
	if info.ParamsOneOf != nil {
		parameters, err := info.ParamsOneOf.ToJSONSchema()
		if err != nil {
			return nil, fmt.Errorf("einoadapter: convert schema for tool %q: %w", info.Name, err)
		}
		definition.Parameters, err = json.Marshal(parameters)
		if err != nil {
			return nil, fmt.Errorf("einoadapter: marshal schema for tool %q: %w", info.Name, err)
		}
	}
	return &Tool{base: base, definition: definition}, nil
}

func (t *Tool) Definition() agentcore.ToolDefinition {
	return t.definition
}

func (t *Tool) Execute(ctx context.Context, arguments json.RawMessage, update agentcore.ToolUpdateSink) (agentcore.ToolResult, error) {
	if streamable, ok := t.base.(tool.EnhancedStreamableTool); ok {
		stream, err := streamable.StreamableRun(ctx, &schema.ToolArgument{Text: string(arguments)})
		if err != nil {
			return agentcore.ToolResult{}, err
		}
		defer stream.Close()
		var content []agentcore.ContentBlock
		var details []*schema.ToolResult
		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return agentcore.ToolResult{Content: content, Details: details}, err
			}
			chunkContent := enhancedResultContent(chunk)
			content = append(content, chunkContent...)
			details = append(details, chunk)
			if update != nil {
				if err := update(agentcore.ToolResult{Content: chunkContent, Details: chunk}); err != nil {
					return agentcore.ToolResult{Content: content, Details: details}, err
				}
			}
		}
		return agentcore.ToolResult{Content: content, Details: details}, nil
	}
	if enhanced, ok := t.base.(tool.EnhancedInvokableTool); ok {
		result, err := enhanced.InvokableRun(ctx, &schema.ToolArgument{Text: string(arguments)})
		if err != nil {
			return agentcore.ToolResult{}, err
		}
		return agentcore.ToolResult{Content: enhancedResultContent(result), Details: result}, nil
	}
	if streamable, ok := t.base.(tool.StreamableTool); ok {
		stream, err := streamable.StreamableRun(ctx, string(arguments))
		if err != nil {
			return agentcore.ToolResult{}, err
		}
		defer stream.Close()
		var output string
		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return agentcore.TextToolResult(output), err
			}
			output += chunk
			if update != nil {
				if err := update(agentcore.TextToolResult(chunk)); err != nil {
					return agentcore.TextToolResult(output), err
				}
			}
		}
		return agentcore.TextToolResult(output), nil
	}
	if invokable, ok := t.base.(tool.InvokableTool); ok {
		output, err := invokable.InvokableRun(ctx, string(arguments))
		return agentcore.TextToolResult(output), err
	}
	return agentcore.ToolResult{}, errors.New("einoadapter: tool is not executable")
}

func enhancedResultContent(result *schema.ToolResult) []agentcore.ContentBlock {
	if result == nil {
		return nil
	}
	content := make([]agentcore.ContentBlock, 0, len(result.Parts))
	for _, part := range result.Parts {
		switch part.Type {
		case schema.ToolPartTypeText:
			content = append(content, agentcore.ContentBlock{Type: agentcore.ContentText, Text: part.Text})
		case schema.ToolPartTypeImage:
			if part.Image != nil {
				content = append(content, commonToCore(agentcore.ContentImage, part.Image.MessagePartCommon, ""))
			}
		case schema.ToolPartTypeAudio:
			if part.Audio != nil {
				content = append(content, commonToCore(agentcore.ContentAudio, part.Audio.MessagePartCommon, ""))
			}
		case schema.ToolPartTypeVideo:
			if part.Video != nil {
				content = append(content, commonToCore(agentcore.ContentVideo, part.Video.MessagePartCommon, ""))
			}
		case schema.ToolPartTypeFile:
			if part.File != nil {
				content = append(content, commonToCore(agentcore.ContentFile, part.File.MessagePartCommon, ""))
			}
		}
	}
	return content
}
