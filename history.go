package agentcore

import (
	"encoding/json"
	"fmt"
)

// HistoryIssueKind identifies a repair or validation problem in model-visible
// conversation history.
type HistoryIssueKind string

const (
	HistoryDroppedFailedAssistant HistoryIssueKind = "dropped_failed_assistant"
	HistoryDroppedInvalidToolCall HistoryIssueKind = "dropped_invalid_tool_call"
	HistoryNormalizedToolCallID   HistoryIssueKind = "normalized_tool_call_id"
	HistoryAddedToolResult        HistoryIssueKind = "added_missing_tool_result"
	HistoryDroppedToolResult      HistoryIssueKind = "dropped_orphan_tool_result"
)

// HistoryIssue describes one deterministic change made by RepairHistory.
type HistoryIssue struct {
	Kind       HistoryIssueKind `json:"kind"`
	Index      int              `json:"index"`
	ToolCallID string           `json:"toolCallId,omitempty"`
	Detail     string           `json:"detail,omitempty"`
}

// HistoryReport summarizes history repair. Changed is false for already-valid
// history, in which case RepairHistory preserves every message byte-for-byte.
type HistoryReport struct {
	Changed bool           `json:"changed"`
	Issues  []HistoryIssue `json:"issues,omitempty"`
}

const interruptedToolResultText = "tool execution was interrupted before its result was durably recorded; its effects are unknown, so inspect current state before retrying"

// RepairHistory returns a detached, provider-safe conversation history. It is
// deterministic, idempotent, and never mutates its input. Failed/aborted
// assistant attempts and malformed partial tool calls are removed; orphan or
// duplicate tool results are dropped; every valid completed tool-call batch is
// immediately followed by exactly one result per call in source order.
func RepairHistory(messages []Message) ([]Message, HistoryReport) {
	repaired := make([]Message, 0, len(messages))
	report := HistoryReport{}
	for index := 0; index < len(messages); {
		message := cloneMessage(messages[index])
		if message.Role == RoleTool {
			report.add(HistoryIssue{Kind: HistoryDroppedToolResult, Index: index, ToolCallID: message.ToolCallID, Detail: "tool result has no immediately preceding tool call"})
			index++
			continue
		}
		if message.Role != RoleAssistant {
			repaired = append(repaired, message)
			index++
			continue
		}
		if message.IsError || message.StopReason == StopReasonError || message.StopReason == StopReasonAborted {
			report.add(HistoryIssue{Kind: HistoryDroppedFailedAssistant, Index: index, Detail: "failed or aborted assistant attempts are diagnostic data, not replay history"})
			index++
			continue
		}

		messageChanged := false
		seenIDs := make(map[string]struct{})
		validCalls := make([]ToolCall, 0)
		validContent := make([]ContentBlock, 0, len(message.Content))
		for blockIndex, block := range message.Content {
			if block.Type != ContentToolCall {
				validContent = append(validContent, block)
				continue
			}
			if block.ToolCall == nil || block.ToolCall.Name == "" || len(block.ToolCall.Arguments) == 0 || !json.Valid(block.ToolCall.Arguments) {
				report.add(HistoryIssue{Kind: HistoryDroppedInvalidToolCall, Index: index, Detail: fmt.Sprintf("invalid or incomplete tool call block %d", blockIndex)})
				messageChanged = true
				continue
			}
			call := cloneToolCall(*block.ToolCall)
			if call.ID == "" {
				call.ID = stableToolCallID(message, index, blockIndex)
				report.add(HistoryIssue{Kind: HistoryNormalizedToolCallID, Index: index, ToolCallID: call.ID, Detail: "missing tool call ID"})
				messageChanged = true
			} else if _, duplicate := seenIDs[call.ID]; duplicate {
				call.ID = stableToolCallID(message, index, blockIndex)
				report.add(HistoryIssue{Kind: HistoryNormalizedToolCallID, Index: index, ToolCallID: call.ID, Detail: "duplicate tool call ID"})
				messageChanged = true
			}
			seenIDs[call.ID] = struct{}{}
			block.ToolCall = &call
			validContent = append(validContent, block)
			validCalls = append(validCalls, call)
		}
		if messageChanged {
			message.Content = validContent
			message.ProviderData = nil
		}
		if len(message.Content) == 0 {
			report.add(HistoryIssue{Kind: HistoryDroppedFailedAssistant, Index: index, Detail: "assistant contained no replayable content"})
			index++
			continue
		}
		repaired = append(repaired, message)
		index++
		if len(validCalls) == 0 {
			continue
		}

		available := make(map[string]Message, len(validCalls))
		observedOrder := make([]string, 0, len(validCalls))
		for index < len(messages) && messages[index].Role == RoleTool {
			result := cloneMessage(messages[index])
			observedOrder = append(observedOrder, result.ToolCallID)
			if _, wanted := seenIDs[result.ToolCallID]; wanted {
				if _, duplicate := available[result.ToolCallID]; !duplicate {
					available[result.ToolCallID] = result
				} else {
					report.add(HistoryIssue{Kind: HistoryDroppedToolResult, Index: index, ToolCallID: result.ToolCallID, Detail: "duplicate tool result"})
				}
			} else {
				report.add(HistoryIssue{Kind: HistoryDroppedToolResult, Index: index, ToolCallID: result.ToolCallID, Detail: "tool result does not match this tool-call batch"})
			}
			index++
		}
		for callIndex, call := range validCalls {
			if callIndex < len(observedOrder) && observedOrder[callIndex] != call.ID {
				report.Changed = true
				break
			}
		}
		for _, call := range validCalls {
			if result, ok := available[call.ID]; ok {
				if result.ToolName != call.Name {
					result.ToolName = call.Name
					report.Changed = true
				}
				repaired = append(repaired, result)
				continue
			}
			result := toolResultMessage(call, errorToolResult(interruptedToolResultText))
			result.ID = stableToolResultID(message, call.ID)
			repaired = append(repaired, result)
			report.add(HistoryIssue{Kind: HistoryAddedToolResult, Index: index - 1, ToolCallID: call.ID, Detail: "missing result; tool effects may be unknown"})
		}
	}
	return repaired, report
}

func (r *HistoryReport) add(issue HistoryIssue) {
	r.Changed = true
	r.Issues = append(r.Issues, issue)
}

func stableToolCallID(message Message, messageIndex, blockIndex int) string {
	prefix := message.ID
	if prefix == "" {
		prefix = fmt.Sprintf("message-%d", messageIndex)
	}
	return fmt.Sprintf("%s-call-%d", prefix, blockIndex)
}

func stableToolResultID(message Message, callID string) string {
	prefix := message.ID
	if prefix == "" {
		prefix = "repaired"
	}
	return prefix + "-result-" + callID
}

// ValidateHistory verifies the provider-neutral replay invariants enforced by
// RepairHistory. It does not mutate messages.
func ValidateHistory(messages []Message) error {
	for index := 0; index < len(messages); index++ {
		message := messages[index]
		if message.Role == RoleTool {
			return fmt.Errorf("agentcore: invalid history at message %d: orphan tool result %q", index, message.ToolCallID)
		}
		if message.Role != RoleAssistant {
			continue
		}
		if message.IsError || message.StopReason == StopReasonError || message.StopReason == StopReasonAborted {
			return fmt.Errorf("agentcore: invalid history at message %d: failed assistant attempt is not replayable", index)
		}
		calls := message.ToolCalls()
		seen := make(map[string]struct{}, len(calls))
		for _, call := range calls {
			if call.ID == "" || call.Name == "" || len(call.Arguments) == 0 || !json.Valid(call.Arguments) {
				return fmt.Errorf("agentcore: invalid history at message %d: incomplete tool call", index)
			}
			if _, duplicate := seen[call.ID]; duplicate {
				return fmt.Errorf("agentcore: invalid history at message %d: duplicate tool call ID %q", index, call.ID)
			}
			seen[call.ID] = struct{}{}
		}
		for callIndex, call := range calls {
			resultIndex := index + 1 + callIndex
			if resultIndex >= len(messages) || messages[resultIndex].Role != RoleTool || messages[resultIndex].ToolCallID != call.ID {
				return fmt.Errorf("agentcore: invalid history at message %d: missing ordered result for tool call %q", index, call.ID)
			}
		}
		index += len(calls)
	}
	return nil
}
