package openai

import "encoding/json"

// mergeReasoningDetails reconstructs logical reasoning blocks from indexed
// deltas. Keep first-seen block order, and concatenate only the protocol's
// streamed payload fields; IDs, types and formats are metadata, not deltas.
// Unindexed, unknown or conflicting blocks stay separate rather than guessing
// which signed/encrypted block they belong to.
func mergeReasoningDetails(existing, deltas []any) []any {
	for _, raw := range deltas {
		delta, ok := raw.(map[string]any)
		index, indexed := reasoningDetailIndex(delta)
		matched := false
		if ok && indexed && knownReasoningType(delta) {
			for position := len(existing) - 1; position >= 0; position-- {
				target, ok := existing[position].(map[string]any)
				targetIndex, indexed := reasoningDetailIndex(target)
				if !ok || !indexed || !knownReasoningType(target) {
					break
				}
				if targetIndex != index {
					continue
				}
				if compatibleReasoningDetails(target, delta) {
					mergeObject(target, delta, "reasoning_details")
					matched = true
				}
				break
			}
		}
		if !matched {
			existing = append(existing, cloneJSONValue(raw))
		}
	}
	return existing
}

func reasoningDetailIndex(detail map[string]any) (int64, bool) {
	// Stream and replay JSON are both decoded with UseNumber. Invalid indices
	// must not silently become zero and merge unrelated blocks.
	number, ok := detail["index"].(json.Number)
	if !ok {
		return 0, false
	}
	index, err := number.Int64()
	return index, err == nil && index >= 0
}

func knownReasoningType(detail map[string]any) bool {
	switch detail["type"] {
	case nil, "", "reasoning.text", "reasoning.summary", "reasoning.encrypted":
		return true
	default:
		return false
	}
}

func compatibleReasoningDetails(target, delta map[string]any) bool {
	for _, field := range []string{"type", "id", "format"} {
		previous, _ := target[field].(string)
		next, _ := delta[field].(string)
		if previous != "" && next != "" && previous != next {
			return false
		}
	}
	return true
}
