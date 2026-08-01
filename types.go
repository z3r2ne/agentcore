package agentcore

import (
	"context"
	"encoding/json"
	"time"
)

// Role identifies the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentType identifies a block within a message.
type ContentType string

const (
	ContentText     ContentType = "text"
	ContentThinking ContentType = "thinking"
	ContentToolCall ContentType = "tool_call"
	ContentImage    ContentType = "image"
	ContentAudio    ContentType = "audio"
	ContentVideo    ContentType = "video"
	ContentFile     ContentType = "file"
)

// StopReason describes why model generation or the agent run stopped.
type StopReason string

const (
	StopReasonStop       StopReason = "stop"
	StopReasonToolUse    StopReason = "tool_use"
	StopReasonLength     StopReason = "length"
	StopReasonError      StopReason = "error"
	StopReasonAborted    StopReason = "aborted"
	StopReasonMaxTurns   StopReason = "max_turns"
	StopReasonTerminated StopReason = "tool_terminated"
)

// Usage contains provider-reported token accounting for one model response.
type Usage struct {
	InputTokens      int `json:"inputTokens,omitempty"`
	OutputTokens     int `json:"outputTokens,omitempty"`
	CacheReadTokens  int `json:"cacheReadTokens,omitempty"`
	CacheWriteTokens int `json:"cacheWriteTokens,omitempty"`
}

// Add returns the component-wise sum of two usage values.
func (u Usage) Add(other Usage) Usage {
	return Usage{
		InputTokens:      u.InputTokens + other.InputTokens,
		OutputTokens:     u.OutputTokens + other.OutputTokens,
		CacheReadTokens:  u.CacheReadTokens + other.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens + other.CacheWriteTokens,
	}
}

// Pricing describes model prices per one million tokens.
type Pricing struct {
	Currency   string  `json:"currency,omitempty"`
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
}

// Cost is a calculated monetary amount in Pricing.Currency.
type Cost struct {
	Amount   float64 `json:"amount"`
	Currency string  `json:"currency,omitempty"`
}

// Cost calculates usage cost from per-million-token prices.
func (u Usage) Cost(pricing Pricing) Cost {
	amount := float64(u.InputTokens)*pricing.Input +
		float64(u.OutputTokens)*pricing.Output +
		float64(u.CacheReadTokens)*pricing.CacheRead +
		float64(u.CacheWriteTokens)*pricing.CacheWrite
	return Cost{Amount: amount / 1_000_000, Currency: pricing.Currency}
}

// ToolCall is a completed model-requested tool invocation. Arguments contains
// the final JSON value accepted by the tool. Streaming, potentially incomplete
// argument text is exposed only through ToolCallDelta.ArgumentsDelta.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// ContentBlock is one text, thinking, tool-call, or multimodal media block.
type ContentBlock struct {
	Type     ContentType `json:"type"`
	Text     string      `json:"text,omitempty"`
	ToolCall *ToolCall   `json:"toolCall,omitempty"`
	URL      string      `json:"url,omitempty"`
	Data     []byte      `json:"data,omitempty"`
	MIMEType string      `json:"mimeType,omitempty"`
	Name     string      `json:"name,omitempty"`
}

// ProviderData keeps an adapter's lossless provider message alongside its
// serializable representation. Runtime is optional and never serialized.
type ProviderData struct {
	Format  string          `json:"format"`
	Data    json.RawMessage `json:"data"`
	Runtime any             `json:"-"`
}

// Message is the provider-neutral message format retained by the loop.
type Message struct {
	ID           string         `json:"id,omitempty"`
	Role         Role           `json:"role"`
	Content      []ContentBlock `json:"content,omitempty"`
	ToolCallID   string         `json:"toolCallId,omitempty"`
	ToolName     string         `json:"toolName,omitempty"`
	StopReason   StopReason     `json:"stopReason,omitempty"`
	Usage        Usage          `json:"usage,omitempty"`
	Error        string         `json:"error,omitempty"`
	IsError      bool           `json:"isError,omitempty"`
	ProviderData *ProviderData  `json:"providerData,omitempty"`
}

// TextMessage constructs a single-block text message.
func TextMessage(role Role, text string) Message {
	return Message{Role: role, Content: []ContentBlock{{Type: ContentText, Text: text}}}
}

// Text concatenates all text blocks in a message.
func (m Message) Text() string {
	var text string
	for _, block := range m.Content {
		if block.Type == ContentText {
			text += block.Text
		}
	}
	return text
}

// ToolCalls returns tool calls in their original model order.
func (m Message) ToolCalls() []ToolCall {
	calls := make([]ToolCall, 0)
	for _, block := range m.Content {
		if block.Type == ContentToolCall && block.ToolCall != nil {
			calls = append(calls, cloneToolCall(*block.ToolCall))
		}
	}
	return calls
}

// ToolDefinition is the schema advertised to a model.
type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ModelRequest contains one complete inference request.
type ModelRequest struct {
	SystemPrompt string           `json:"systemPrompt,omitempty"`
	Messages     []Message        `json:"messages"`
	Tools        []ToolDefinition `json:"tools,omitempty"`
	Options      map[string]any   `json:"options,omitempty"`
}

// ToolCallDelta incrementally builds one tool call. Index is stable within a
// single assistant response. ArgumentsDelta is arbitrary text and is not
// required to be valid JSON until the assistant message is complete.
type ToolCallDelta struct {
	Index          int    `json:"index"`
	ID             string `json:"id,omitempty"`
	Name           string `json:"name,omitempty"`
	ArgumentsDelta string `json:"argumentsDelta,omitempty"`
}

// ModelChunk is one normalized streaming update from a model adapter.
type ModelChunk struct {
	TextDelta      string          `json:"textDelta,omitempty"`
	ThinkingDelta  string          `json:"thinkingDelta,omitempty"`
	ContentDeltas  []ContentBlock  `json:"contentDeltas,omitempty"`
	ToolCallDeltas []ToolCallDelta `json:"toolCallDeltas,omitempty"`
	StopReason     StopReason      `json:"stopReason,omitempty"`
	Usage          *Usage          `json:"usage,omitempty"`
	ProviderData   *ProviderData   `json:"providerData,omitempty"`
}

// ModelStream yields chunks until io.EOF and must be closed by the loop.
type ModelStream interface {
	Recv() (ModelChunk, error)
	Close() error
}

// Model starts a streaming model request.
type Model interface {
	Stream(context.Context, ModelRequest) (ModelStream, error)
}

// ToolExecutionMode controls execution of a batch of tool calls.
type ToolExecutionMode string

const (
	ToolExecutionParallel   ToolExecutionMode = "parallel"
	ToolExecutionSequential ToolExecutionMode = "sequential"
)

// ToolResult is sent back to the model after a tool invocation.
type ToolResult struct {
	Content   []ContentBlock `json:"content,omitempty"`
	Details   any            `json:"details,omitempty"`
	IsError   bool           `json:"isError,omitempty"`
	Terminate bool           `json:"terminate,omitempty"`
}

// TextToolResult constructs a plain-text tool result.
func TextToolResult(text string) ToolResult {
	return ToolResult{Content: []ContentBlock{{Type: ContentText, Text: text}}}
}

// Text concatenates all text blocks in a tool result.
func (r ToolResult) Text() string {
	var text string
	for _, block := range r.Content {
		if block.Type == ContentText {
			text += block.Text
		}
	}
	return text
}

// ToolUpdateSink reports partial tool output.
type ToolUpdateSink func(ToolResult) error

// Tool is executable model-facing functionality.
type Tool interface {
	Definition() ToolDefinition
	Execute(context.Context, json.RawMessage, ToolUpdateSink) (ToolResult, error)
}

// ToolValidator optionally validates arguments before a tool starts.
type ToolValidator interface {
	Validate(json.RawMessage) error
}

// ToolExecutionModeProvider can force all calls in a batch to run sequentially.
type ToolExecutionModeProvider interface {
	ExecutionMode() ToolExecutionMode
}

// State is the reusable in-memory state of an agent session.
type State struct {
	Messages []Message `json:"messages"`
}

// Result is returned after a complete or interrupted run.
type Result struct {
	State       State      `json:"state"`
	NewMessages []Message  `json:"newMessages"`
	Turns       int        `json:"turns"`
	StopReason  StopReason `json:"stopReason"`
	Usage       Usage      `json:"usage"`
}

// TurnContext is passed to hooks after a model turn and its tools complete.
type TurnContext struct {
	Turn        int
	Message     Message
	ToolResults []Message
	State       *State
	Usage       Usage
	Next        *NextTurnConfig
}

// NextTurnConfig applies run-local configuration changes before the next model
// turn. A nil field keeps the current value; Tools replaces the complete set.
type NextTurnConfig struct {
	Model         Model
	SystemPrompt  *string
	Tools         *[]Tool
	ToolExecution ToolExecutionMode
	ModelOptions  map[string]any
}

// ToolCallContext describes a tool invocation to a hook.
type ToolCallContext struct {
	Turn     int
	Call     ToolCall
	State    *State
	Executed bool
	Attempts int
}

// ToolCallDecision can block a call or replace its arguments.
type ToolCallDecision struct {
	Block     bool
	Reason    string
	Arguments json.RawMessage
}

// Hooks customize lifecycle behavior without replacing the loop.
type Hooks struct {
	BeforeModelCall func(context.Context, *ModelRequest) error
	AfterModelCall  func(context.Context, *Message) error
	BeforeToolCall  func(context.Context, ToolCallContext) (ToolCallDecision, error)
	AfterToolCall   func(context.Context, ToolCallContext, *ToolResult) error
	PrepareNextTurn func(context.Context, *TurnContext) error
	ShouldStop      func(context.Context, TurnContext) bool
}

// RetryPolicy controls retry of model transport/stream failures. MaxAttempts
// includes the initial attempt. Zero selects one attempt with no retry.
type RetryPolicy struct {
	MaxAttempts  int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
	ShouldRetry  func(error) bool
}

// ToolPolicy controls execution safety for every tool. A tool may override it
// by implementing ToolPolicyProvider.
type ToolPolicy struct {
	Timeout              time.Duration
	MaxAttempts          int
	RetryDelay           time.Duration
	ShouldRetry          func(error) bool
	DisablePanicRecovery bool
}

// ToolPolicyProvider supplies per-tool timeout, retry, and panic behavior.
type ToolPolicyProvider interface {
	ToolPolicy() ToolPolicy
}

// ContextPolicy applies request-only context budgeting. Compact may summarize
// messages; when nil, agentcore keeps the newest complete message suffix.
type ContextPolicy struct {
	MaxTokens          int
	ReserveTokens      int
	MaxToolResultBytes int
	EstimateTokens     func([]Message) int
	Compact            func(context.Context, []Message, int) ([]Message, error)
}

// Config controls one Agent instance.
type Config struct {
	Model              Model
	SystemPrompt       string
	Tools              []Tool
	MaxTurns           int
	ToolExecution      ToolExecutionMode
	MaxToolConcurrency int
	ModelRetry         RetryPolicy
	ToolPolicy         ToolPolicy
	ModelOptions       map[string]any
	ContextPolicy      ContextPolicy
	TransformContext   func(context.Context, []Message) ([]Message, error)
	Hooks              Hooks
	Interceptors       []Interceptor
}

// EventType identifies an observable agent lifecycle event.
type EventType string

const (
	EventAgentStart          EventType = "agent_start"
	EventAgentEnd            EventType = "agent_end"
	EventTurnStart           EventType = "turn_start"
	EventTurnEnd             EventType = "turn_end"
	EventMessageStart        EventType = "message_start"
	EventMessageUpdate       EventType = "message_update"
	EventMessageEnd          EventType = "message_end"
	EventToolExecutionStart  EventType = "tool_execution_start"
	EventToolExecutionUpdate EventType = "tool_execution_update"
	EventToolExecutionEnd    EventType = "tool_execution_end"
	EventAutoRetryStart      EventType = "auto_retry_start"
	EventAutoRetryEnd        EventType = "auto_retry_end"
	EventContextCompactStart EventType = "context_compaction_start"
	EventContextCompactEnd   EventType = "context_compaction_end"
)

// Event mirrors the useful portion of Pi's event stream while remaining a
// typed Go value suitable for callbacks, SSE, WebSockets, or JSONL RPC.
type Event struct {
	Type         EventType       `json:"type"`
	Turn         int             `json:"turn,omitempty"`
	Message      *Message        `json:"message,omitempty"`
	Delta        *ModelChunk     `json:"delta,omitempty"`
	ToolCallID   string          `json:"toolCallId,omitempty"`
	ToolName     string          `json:"toolName,omitempty"`
	Arguments    json.RawMessage `json:"args,omitempty"`
	ToolResult   *ToolResult     `json:"result,omitempty"`
	IsError      bool            `json:"isError,omitempty"`
	Error        string          `json:"error,omitempty"`
	Messages     []Message       `json:"messages,omitempty"`
	ToolResults  []Message       `json:"toolResults,omitempty"`
	Attempt      int             `json:"attempt,omitempty"`
	MaxAttempts  int             `json:"maxAttempts,omitempty"`
	Delay        time.Duration   `json:"delay,omitempty"`
	Success      bool            `json:"success,omitempty"`
	BeforeTokens int             `json:"beforeTokens,omitempty"`
	AfterTokens  int             `json:"afterTokens,omitempty"`
}

// EventSink receives events synchronously and in deterministic order.
type EventSink func(context.Context, Event) error

func cloneToolCall(call ToolCall) ToolCall {
	call.Arguments = append(json.RawMessage(nil), call.Arguments...)
	return call
}
