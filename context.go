package agentcore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

// SummaryCompactorConfig configures a model-generated context summary.
type SummaryCompactorConfig struct {
	Model          Model
	SystemPrompt   string
	KeepRecent     int
	EstimateTokens func([]Message) int
}

// NewSummaryCompactor returns a ContextPolicy.Compact implementation that
// summarizes older messages and preserves a recent suffix.
func NewSummaryCompactor(config SummaryCompactorConfig) (func(context.Context, []Message, int) ([]Message, error), error) {
	if config.Model == nil {
		return nil, ErrModelRequired
	}
	if config.KeepRecent <= 0 {
		config.KeepRecent = 4
	}
	if config.SystemPrompt == "" {
		config.SystemPrompt = "Summarize the conversation for another assistant. Preserve decisions, constraints, file paths, errors, tool results, and unfinished work. Return only the summary."
	}
	if config.EstimateTokens == nil {
		config.EstimateTokens = EstimateTokens
	}
	return func(ctx context.Context, messages []Message, target int) ([]Message, error) {
		if len(messages) <= config.KeepRecent {
			return trimContext(messages, target, config.EstimateTokens), nil
		}
		split := len(messages) - config.KeepRecent
		for split > 0 && messages[split].Role == RoleTool {
			split--
		}
		older := cloneMessages(messages[:split])
		recent := cloneMessages(messages[split:])
		if len(older) == 0 {
			return trimContext(recent, target, config.EstimateTokens), nil
		}
		stream, err := config.Model.Stream(ctx, ModelRequest{SystemPrompt: config.SystemPrompt, Messages: older})
		if err != nil {
			return nil, fmt.Errorf("start summary model: %w", err)
		}
		if stream == nil {
			return nil, errors.New("summary model returned a nil stream")
		}
		defer stream.Close()
		assistant := Message{Role: RoleAssistant}
		accumulator := responseAccumulator{message: &assistant, toolBlockIndexes: map[int]int{}}
		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return nil, fmt.Errorf("receive summary model: %w", err)
			}
			accumulator.add(chunk)
		}
		if len(assistant.ToolCalls()) > 0 {
			return nil, errors.New("summary model returned a tool call")
		}
		summary := strings.TrimSpace(assistant.Text())
		if summary == "" {
			return nil, errors.New("summary model returned empty text")
		}
		compacted := append([]Message{TextMessage(RoleUser, "[Earlier conversation summary]\n"+summary)}, recent...)
		if config.EstimateTokens(compacted) > target {
			compacted = trimContext(compacted, target, config.EstimateTokens)
		}
		return compacted, nil
	}, nil
}

func (a *Agent) applyContextPolicy(ctx context.Context, messages []Message, turn int, emit *eventEmitter) ([]Message, error) {
	policy := a.config.ContextPolicy
	messages = cloneMessages(messages)
	truncated := truncateToolResults(messages, policy.MaxToolResultBytes)
	if policy.MaxTokens <= 0 {
		return messages, nil
	}
	target := policy.MaxTokens - policy.ReserveTokens
	if target <= 0 {
		return nil, errors.New("agentcore: context reserve must be smaller than max tokens")
	}
	estimate := policy.EstimateTokens
	if estimate == nil {
		estimate = EstimateTokens
	}
	before := estimate(messages)
	if before <= target && !truncated {
		return messages, nil
	}
	if err := emit.send(Event{Type: EventContextCompactStart, Turn: turn, BeforeTokens: before}); err != nil {
		return nil, err
	}
	var compacted []Message
	var err error
	if before > target {
		if policy.Compact != nil {
			compacted, err = policy.Compact(ctx, cloneMessages(messages), target)
		} else {
			compacted = trimContext(messages, target, estimate)
		}
		if err != nil {
			wrapped := fmt.Errorf("compact context: %w", err)
			_ = emit.send(Event{Type: EventContextCompactEnd, Turn: turn, BeforeTokens: before, IsError: true, Error: wrapped.Error()})
			return nil, wrapped
		}
	} else {
		compacted = messages
	}
	after := estimate(compacted)
	if after > target {
		err := fmt.Errorf("agentcore: compacted context uses %d tokens, target is %d", after, target)
		_ = emit.send(Event{Type: EventContextCompactEnd, Turn: turn, BeforeTokens: before, AfterTokens: after, IsError: true, Error: err.Error()})
		return nil, err
	}
	if err := emit.send(Event{Type: EventContextCompactEnd, Turn: turn, BeforeTokens: before, AfterTokens: after, Success: true}); err != nil {
		return nil, err
	}
	return compacted, nil
}

// EstimateTokens provides a deterministic provider-neutral approximation.
// Provider adapters can replace it with their model tokenizer.
func EstimateTokens(messages []Message) int {
	visible := cloneMessages(messages)
	for index := range visible {
		visible[index].ID = ""
		visible[index].Usage = Usage{}
		visible[index].ProviderData = nil
	}
	data, err := json.Marshal(visible)
	if err != nil {
		return 0
	}
	// Four UTF-8 bytes per token is a conservative general-purpose estimate.
	return (len(data) + 3) / 4
}

func trimContext(messages []Message, target int, estimate func([]Message) int) []Message {
	if len(messages) == 0 {
		return nil
	}
	start := len(messages) - 1
	for start > 0 {
		candidate := messages[start-1:]
		if estimate(candidate) > target {
			break
		}
		start--
	}
	// A tool result cannot stand without the assistant tool call that introduced
	// it. Drop orphaned leading results rather than creating an invalid request.
	for start < len(messages) && messages[start].Role == RoleTool {
		start++
	}
	if start == len(messages) {
		return nil
	}
	result := cloneMessages(messages[start:])
	for len(result) > 1 && estimate(result) > target {
		result = result[1:]
	}
	return result
}

func truncateToolResults(messages []Message, maxBytes int) bool {
	if maxBytes <= 0 {
		return false
	}
	changed := false
	for messageIndex := range messages {
		if messages[messageIndex].Role != RoleTool {
			continue
		}
		for blockIndex := range messages[messageIndex].Content {
			block := &messages[messageIndex].Content[blockIndex]
			if block.Type == ContentText && len(block.Text) > maxBytes {
				block.Text = truncateUTF8(block.Text, maxBytes)
				changed = true
			}
			if len(block.Data) > maxBytes || len(block.URL) > maxBytes {
				originalBytes := len(block.Data) + len(block.URL)
				kind := block.Type
				message := fmt.Sprintf("[%s tool result omitted: %d bytes exceeds %d-byte limit]", kind, originalBytes, maxBytes)
				*block = ContentBlock{
					Type: ContentText,
					Text: truncateUTF8Prefix(message, maxBytes),
				}
				changed = true
			}
		}
	}
	return changed
}

func truncateUTF8Prefix(value string, maxBytes int) string {
	if len(value) <= maxBytes {
		return value
	}
	value = value[:maxBytes]
	for !utf8.ValidString(value) {
		value = value[:len(value)-1]
	}
	return value
}

func truncateUTF8(value string, maxBytes int) string {
	const marker = "\n... tool result truncated ...\n"
	if len(value) <= maxBytes {
		return value
	}
	if maxBytes <= len(marker) {
		return marker[:maxBytes]
	}
	remaining := maxBytes - len(marker)
	leftBytes := remaining / 2
	rightBytes := remaining - leftBytes
	left := value[:leftBytes]
	for !utf8.ValidString(left) {
		left = left[:len(left)-1]
	}
	right := value[len(value)-rightBytes:]
	for !utf8.ValidString(right) {
		right = right[1:]
	}
	return left + marker + right
}
