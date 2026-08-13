package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"

	"github.com/z3r2ne/agentcore"
)

type limitedReadCloser struct {
	body      io.ReadCloser
	remaining int64
	once      sync.Once
}

func (r *limitedReadCloser) Read(p []byte) (int, error) {
	if r.remaining <= 0 {
		return 0, errors.New("response body exceeds configured limit")
	}
	if int64(len(p)) > r.remaining {
		p = p[:r.remaining]
	}
	n, e := r.body.Read(p)
	r.remaining -= int64(n)
	return n, e
}
func (r *limitedReadCloser) Close() error {
	var e error
	r.once.Do(func() { e = r.body.Close() })
	return e
}

type stream struct {
	ctx          context.Context
	body         io.ReadCloser
	scanner      *bufio.Scanner
	max          int
	event        string
	data         []byte
	done, closed bool
	blocks       map[int]map[string]any
	order        []int
	sawStop      bool
	inputUsage   agentcore.Usage
}

func newStream(ctx context.Context, body io.ReadCloser, max int) *stream {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 64<<10), max+1024)
	return &stream{ctx: ctx, body: body, scanner: sc, max: max, blocks: map[int]map[string]any{}}
}

func (s *stream) Recv() (agentcore.ModelChunk, error) {
	if s.done || s.closed {
		return agentcore.ModelChunk{}, io.EOF
	}
	for {
		event, data, err := s.next()
		if err != nil {
			s.done = true
			if errors.Is(err, io.EOF) && s.sawStop {
				return agentcore.ModelChunk{}, io.EOF
			}
			if s.ctx.Err() != nil {
				return agentcore.ModelChunk{}, s.ctx.Err()
			}
			return agentcore.ModelChunk{}, &Error{Operation: "read stream", Message: "stream ended before message_stop", Retryable: true, Err: io.ErrUnexpectedEOF}
		}
		chunk, skip, err := s.decode(event, data)
		if err != nil {
			s.done = true
			return agentcore.ModelChunk{}, err
		}
		if !skip {
			return chunk, nil
		}
	}
}

func (s *stream) next() (string, []byte, error) {
	for s.scanner.Scan() {
		line := s.scanner.Bytes()
		if len(line) == 0 {
			if len(s.data) == 0 {
				continue
			}
			event := s.event
			data := append([]byte(nil), s.data...)
			s.event = ""
			s.data = nil
			return event, data, nil
		}
		field, value, ok := bytes.Cut(line, []byte(":"))
		if !ok {
			continue
		}
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}
		switch string(field) {
		case "event":
			s.event = string(value)
		case "data":
			if len(s.data) > 0 {
				s.data = append(s.data, '\n')
			}
			if len(s.data)+len(value) > s.max {
				return "", nil, errors.New("SSE event exceeds configured limit")
			}
			s.data = append(s.data, value...)
		}
	}
	if err := s.scanner.Err(); err != nil {
		return "", nil, err
	}
	return "", nil, io.EOF
}

func (s *stream) decode(event string, data []byte) (agentcore.ModelChunk, bool, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return agentcore.ModelChunk{}, false, &Error{Operation: "decode event", Retryable: true, Err: err}
	}
	if event == "error" || raw["error"] != nil {
		var envelope struct {
			Error struct{ Type, Message string }
		}
		_ = json.Unmarshal(data, &envelope)
		return agentcore.ModelChunk{}, false, &Error{Operation: "stream", Type: envelope.Error.Type, Message: envelope.Error.Message, Retryable: strings.Contains(envelope.Error.Type, "overload") || strings.Contains(envelope.Error.Type, "rate")}
	}
	chunk := agentcore.ModelChunk{}
	switch event {
	case "message_start":
		var value struct {
			Message struct {
				Usage struct {
					Input, CacheCreation, CacheRead int `json:"-"`
				} `json:"usage"`
			} `json:"message"`
		}
		var generic struct {
			Message struct {
				Usage struct {
					Input         int `json:"input_tokens"`
					CacheCreation int `json:"cache_creation_input_tokens"`
					CacheRead     int `json:"cache_read_input_tokens"`
				} `json:"usage"`
			} `json:"message"`
		}
		_ = value
		_ = json.Unmarshal(data, &generic)
		s.inputUsage = agentcore.Usage{InputTokens: max(0, generic.Message.Usage.Input-generic.Message.Usage.CacheRead-generic.Message.Usage.CacheCreation), CacheReadTokens: generic.Message.Usage.CacheRead, CacheWriteTokens: generic.Message.Usage.CacheCreation}
		return chunk, true, nil
	case "content_block_start":
		var value struct {
			Index   int            `json:"index"`
			Content map[string]any `json:"content_block"`
		}
		if err := json.Unmarshal(data, &value); err != nil {
			return chunk, false, err
		}
		s.blocks[value.Index] = value.Content
		s.order = append(s.order, value.Index)
		kind, _ := value.Content["type"].(string)
		if kind == "tool_use" {
			id, _ := value.Content["id"].(string)
			name, _ := value.Content["name"].(string)
			chunk.ToolCallDeltas = []agentcore.ToolCallDelta{{Index: value.Index, ID: id, Name: name}}
		}
		return chunk, len(chunk.ToolCallDeltas) == 0, nil
	case "content_block_delta":
		var value struct {
			Index int `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Thinking    string `json:"thinking"`
				Signature   string `json:"signature"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(data, &value); err != nil {
			return chunk, false, err
		}
		block := s.blocks[value.Index]
		switch value.Delta.Type {
		case "text_delta":
			chunk.TextDelta = value.Delta.Text
			block["text"], _ = appendString(block["text"], value.Delta.Text)
		case "thinking_delta":
			chunk.ThinkingDelta = value.Delta.Thinking
			block["thinking"], _ = appendString(block["thinking"], value.Delta.Thinking)
		case "signature_delta":
			block["signature"], _ = appendString(block["signature"], value.Delta.Signature)
		case "input_json_delta":
			chunk.ToolCallDeltas = []agentcore.ToolCallDelta{{Index: value.Index, ArgumentsDelta: value.Delta.PartialJSON}}
			block["partial_json"], _ = appendString(block["partial_json"], value.Delta.PartialJSON)
		}
		return chunk, chunk.TextDelta == "" && chunk.ThinkingDelta == "" && len(chunk.ToolCallDeltas) == 0, nil
	case "content_block_stop":
		return chunk, true, nil
	case "message_delta":
		var value struct {
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage struct {
				Output int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(data, &value); err != nil {
			return chunk, false, err
		}
		chunk.StopReason = anthropicStop(value.Delta.StopReason)
		usage := s.inputUsage
		usage.OutputTokens = value.Usage.Output
		chunk.Usage = &usage
		provider, err := s.providerData()
		if err != nil {
			return chunk, false, err
		}
		chunk.ProviderData = provider
		return chunk, false, nil
	case "message_stop":
		s.sawStop = true
		s.done = true
		return chunk, true, io.EOF
	case "ping":
		return chunk, true, nil
	default:
		return chunk, true, nil
	}
}

func appendString(existing any, delta string) (string, bool) {
	value, _ := existing.(string)
	return value + delta, true
}
func (s *stream) providerData() (*agentcore.ProviderData, error) {
	content := make([]any, 0, len(s.order))
	for _, index := range s.order {
		block := s.blocks[index]
		if partial, ok := block["partial_json"].(string); ok {
			var input any
			if json.Unmarshal([]byte(partial), &input) == nil {
				block["input"] = input
			}
			delete(block, "partial_json")
		}
		content = append(content, block)
	}
	data, err := json.Marshal(content)
	if err != nil {
		return nil, err
	}
	return &agentcore.ProviderData{Format: ProviderDataFormat, Data: data}, nil
}
func anthropicStop(reason string) agentcore.StopReason {
	switch reason {
	case "end_turn", "stop_sequence", "pause_turn":
		return agentcore.StopReasonStop
	case "tool_use":
		return agentcore.StopReasonToolUse
	case "max_tokens":
		return agentcore.StopReasonLength
	case "refusal":
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
	return s.body.Close()
}

var _ agentcore.ModelStream = (*stream)(nil)
