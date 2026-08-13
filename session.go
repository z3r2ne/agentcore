package agentcore

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// DeliveryMode controls whether one or all queued messages are injected into
// the next available turn.
type DeliveryMode string

const (
	DeliveryOne DeliveryMode = "one-at-a-time"
	DeliveryAll DeliveryMode = "all"
)

// SessionOptions configures queue delivery for a stateful Session.
type SessionOptions struct {
	SteeringMode DeliveryMode
	FollowUpMode DeliveryMode
	// Store and SessionID enable automatic checkpoints after completed runs and
	// whenever a queued message is accepted.
	Store     SessionStore
	SessionID string
}

// SessionStore is implemented by optional persistence adapters such as
// agentcore/sqlitestore. Implementations must save snapshots atomically.
type SessionStore interface {
	SaveSession(context.Context, string, SessionSnapshot) error
}

// SessionSnapshot is the durable portion of a Session. Active streams, tools,
// subscribers, and cancellation functions are intentionally absent.
type SessionSnapshot struct {
	State        State        `json:"state"`
	Usage        Usage        `json:"usage"`
	Steering     []Message    `json:"steering,omitempty"`
	FollowUp     []Message    `json:"followUp,omitempty"`
	SteeringMode DeliveryMode `json:"steeringMode"`
	FollowUpMode DeliveryMode `json:"followUpMode"`
	LastError    string       `json:"lastError,omitempty"`
}

// SessionStatus is a concurrency-safe snapshot of an active session.
type SessionStatus struct {
	Running          bool
	StreamingMessage *Message
	PendingToolCalls map[string]string
	SteeringQueued   int
	FollowUpQueued   int
	LastError        string
	Usage            Usage
}

// Session wraps Agent with durable in-memory state, runtime observation,
// cancellation, and Pi-style steering/follow-up queues.
type Session struct {
	agent *Agent

	mu               sync.RWMutex
	state            State
	options          SessionOptions
	running          bool
	cancel           context.CancelFunc
	idle             chan struct{}
	streaming        *Message
	pendingTools     map[string]string
	steering         []Message
	followUp         []Message
	lastError        string
	usage            Usage
	subscribers      []sessionSubscriber
	nextSubscriberID uint64
}

type sessionSubscriber struct {
	id   uint64
	sink EventSink
}

// NewSession creates a stateful wrapper around agent.
func NewSession(agent *Agent, initial State, options SessionOptions) (*Session, error) {
	if agent == nil {
		return nil, errors.New("agentcore: nil agent")
	}
	if options.SteeringMode == "" {
		options.SteeringMode = DeliveryOne
	}
	if options.FollowUpMode == "" {
		options.FollowUpMode = DeliveryOne
	}
	if options.SteeringMode != DeliveryOne && options.SteeringMode != DeliveryAll {
		return nil, errors.New("agentcore: invalid steering delivery mode")
	}
	if options.FollowUpMode != DeliveryOne && options.FollowUpMode != DeliveryAll {
		return nil, errors.New("agentcore: invalid follow-up delivery mode")
	}
	if options.Store != nil && options.SessionID == "" {
		return nil, errors.New("agentcore: session ID is required when a store is configured")
	}
	idle := make(chan struct{})
	close(idle)
	repaired, _ := RepairHistory(initial.Messages)
	return &Session{
		agent: agent, state: State{Messages: repaired}, options: options,
		idle: idle, pendingTools: map[string]string{},
	}, nil
}

// NewSessionFromSnapshot restores durable state and queues. Runtime options
// such as Store and SessionID are supplied separately; zero delivery modes use
// the modes recorded in the snapshot.
func NewSessionFromSnapshot(agent *Agent, snapshot SessionSnapshot, options SessionOptions) (*Session, error) {
	if options.SteeringMode == "" {
		options.SteeringMode = snapshot.SteeringMode
	}
	if options.FollowUpMode == "" {
		options.FollowUpMode = snapshot.FollowUpMode
	}
	// A process can stop after an assistant tool-call message is checkpointed
	// but before every tool result is committed. Repair that interrupted turn
	// before accepting new prompts so OpenAI-compatible providers do not reject
	// the restored history as an invalid tool-call sequence.
	snapshot.State.Messages, _ = RepairHistory(snapshot.State.Messages)
	session, err := NewSession(agent, snapshot.State, options)
	if err != nil {
		return nil, err
	}
	session.steering = cloneMessages(snapshot.Steering)
	session.followUp = cloneMessages(snapshot.FollowUp)
	session.usage = snapshot.Usage
	session.lastError = snapshot.LastError
	return session, nil
}

// Prompt runs prompts against the current session state. Only one run may be
// active; use Steer or FollowUp to add messages to an active run.
func (s *Session) Prompt(ctx context.Context, prompts []Message, sink EventSink) (Result, error) {
	runCtx, state, prompts, cancel, err := s.begin(ctx, prompts)
	if err != nil {
		return Result{}, err
	}
	return s.execute(runCtx, state, prompts, cancel, sink)
}

func (s *Session) begin(ctx context.Context, prompts []Message) (context.Context, State, []Message, context.CancelFunc, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return nil, State{}, nil, nil, ErrSessionBusy
	}
	// A message accepted at the very end of the previous run must not be lost.
	queued := append(cloneMessages(s.steering), cloneMessages(s.followUp)...)
	s.steering = nil
	s.followUp = nil
	prompts = append(queued, cloneMessages(prompts)...)
	state := State{Messages: cloneMessages(s.state.Messages)}
	runCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	s.running = true
	s.idle = make(chan struct{})
	s.streaming = nil
	s.pendingTools = map[string]string{}
	s.lastError = ""
	s.mu.Unlock()
	return runCtx, state, prompts, cancel, nil
}

func (s *Session) execute(runCtx context.Context, state State, prompts []Message, cancel context.CancelFunc, sink EventSink) (result Result, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("agentcore: session run panicked: %v", recovered)
			if len(result.State.Messages) == 0 {
				result = resultFrom(state, nil, 0, StopReasonError)
			}
		}
		cancel()
		s.mu.Lock()
		s.state = State{Messages: cloneMessages(result.State.Messages)}
		s.usage = s.usage.Add(result.Usage)
		s.running = false
		s.cancel = nil
		s.streaming = nil
		s.pendingTools = map[string]string{}
		if err != nil {
			s.lastError = err.Error()
		}
		if s.options.Store != nil {
			if saveErr := saveSessionSnapshot(context.WithoutCancel(runCtx), s.options.Store, s.options.SessionID, s.snapshotLocked()); saveErr != nil {
				err = errors.Join(err, fmt.Errorf("agentcore: save session checkpoint: %w", saveErr))
				s.lastError = err.Error()
			}
		}
		close(s.idle)
		s.mu.Unlock()
	}()
	result, err = s.agent.prompt(runCtx, state, prompts, s.observe(sink), s)
	return result, err
}

// Stream starts an asynchronous session prompt.
func (s *Session) Stream(ctx context.Context, prompts []Message) *EventStream {
	runCtx, state, preparedPrompts, cancel, err := s.begin(ctx, prompts)
	if err != nil {
		return newEventStream(func(EventSink) (Result, error) { return Result{}, err })
	}
	return newEventStream(func(sink EventSink) (Result, error) {
		return s.execute(runCtx, state, preparedPrompts, cancel, sink)
	})
}

// Continue resumes from the current committed state without a new message.
func (s *Session) Continue(ctx context.Context, sink EventSink) (Result, error) {
	s.mu.RLock()
	if s.running {
		s.mu.RUnlock()
		return Result{}, ErrSessionBusy
	}
	if len(s.state.Messages) == 0 {
		s.mu.RUnlock()
		return Result{}, errors.New("agentcore: cannot continue empty session")
	}
	if s.state.Messages[len(s.state.Messages)-1].Role == RoleAssistant {
		s.mu.RUnlock()
		return Result{}, errors.New("agentcore: cannot continue session from assistant message")
	}
	s.mu.RUnlock()
	return s.Prompt(ctx, nil, sink)
}

// Subscribe registers a persistent, ordered event listener. The returned
// function is idempotent and removes the listener.
func (s *Session) Subscribe(sink EventSink) func() {
	if sink == nil {
		return func() {}
	}
	s.mu.Lock()
	s.nextSubscriberID++
	id := s.nextSubscriberID
	s.subscribers = append(s.subscribers, sessionSubscriber{id: id, sink: sink})
	s.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			s.mu.Lock()
			for index, subscriber := range s.subscribers {
				if subscriber.id == id {
					copy(s.subscribers[index:], s.subscribers[index+1:])
					s.subscribers = s.subscribers[:len(s.subscribers)-1]
					break
				}
			}
			s.mu.Unlock()
		})
	}
}

// Steer injects messages after the current model turn and its tool calls.
func (s *Session) Steer(messages ...Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.running {
		return ErrSessionIdle
	}
	previousLength := len(s.steering)
	s.steering = append(s.steering, cloneMessages(messages)...)
	if s.options.Store != nil {
		if err := saveSessionSnapshot(context.Background(), s.options.Store, s.options.SessionID, s.snapshotLocked()); err != nil {
			s.steering = s.steering[:previousLength]
			return fmt.Errorf("agentcore: save steering checkpoint: %w", err)
		}
	}
	return nil
}

// FollowUp queues messages for delivery when the agent would otherwise stop.
func (s *Session) FollowUp(messages ...Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.running {
		return ErrSessionIdle
	}
	previousLength := len(s.followUp)
	s.followUp = append(s.followUp, cloneMessages(messages)...)
	if s.options.Store != nil {
		if err := saveSessionSnapshot(context.Background(), s.options.Store, s.options.SessionID, s.snapshotLocked()); err != nil {
			s.followUp = s.followUp[:previousLength]
			return fmt.Errorf("agentcore: save follow-up checkpoint: %w", err)
		}
	}
	return nil
}

// Abort cancels the active model stream and tools.
func (s *Session) Abort() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if !s.running || s.cancel == nil {
		return ErrSessionIdle
	}
	s.cancel()
	return nil
}

// WaitForIdle blocks until the active run settles.
func (s *Session) WaitForIdle(ctx context.Context) error {
	s.mu.RLock()
	idle := s.idle
	running := s.running
	s.mu.RUnlock()
	if !running {
		return nil
	}
	select {
	case <-idle:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// State returns a detached copy of committed conversation state.
func (s *Session) State() State {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return State{Messages: cloneMessages(s.state.Messages)}
}

// Snapshot returns a detached, serializable checkpoint. A running session
// cannot be snapshotted externally because its current turn is not committed.
func (s *Session) Snapshot() (SessionSnapshot, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.running {
		return SessionSnapshot{}, ErrSessionBusy
	}
	return s.snapshotLocked(), nil
}

func (s *Session) snapshotLocked() SessionSnapshot {
	return SessionSnapshot{
		State: State{Messages: cloneMessages(s.state.Messages)},
		Usage: s.usage, Steering: cloneMessages(s.steering), FollowUp: cloneMessages(s.followUp),
		SteeringMode: s.options.SteeringMode, FollowUpMode: s.options.FollowUpMode,
		LastError: s.lastError,
	}
}

// Status returns a detached runtime snapshot.
func (s *Session) Status() SessionStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()
	status := SessionStatus{
		Running: s.running, PendingToolCalls: make(map[string]string, len(s.pendingTools)),
		SteeringQueued: len(s.steering), FollowUpQueued: len(s.followUp),
		LastError: s.lastError, Usage: s.usage,
	}
	if s.streaming != nil {
		copy := cloneMessage(*s.streaming)
		status.StreamingMessage = &copy
	}
	for id, name := range s.pendingTools {
		status.PendingToolCalls[id] = name
	}
	return status
}

func (s *Session) observe(next EventSink) EventSink {
	return func(ctx context.Context, event Event) error {
		s.mu.Lock()
		switch event.Type {
		case EventMessageStart, EventMessageUpdate:
			if event.Message != nil && event.Message.Role == RoleAssistant {
				copy := cloneMessage(*event.Message)
				s.streaming = &copy
			}
		case EventMessageEnd:
			if event.Message != nil && event.Message.Role == RoleAssistant {
				s.streaming = nil
			}
		case EventToolExecutionStart:
			s.pendingTools[event.ToolCallID] = event.ToolName
		case EventToolExecutionEnd:
			delete(s.pendingTools, event.ToolCallID)
		case EventAgentEnd:
			s.streaming = nil
		}
		subscribers := append([]sessionSubscriber(nil), s.subscribers...)
		s.mu.Unlock()
		if next != nil {
			if err := next(ctx, event); err != nil {
				return err
			}
		}
		for _, subscriber := range subscribers {
			if err := subscriber.sink(ctx, event); err != nil {
				return err
			}
		}
		return nil
	}
}

func (s *Session) takeSteering() []Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return takeQueued(&s.steering, s.options.SteeringMode)
}

func (s *Session) takeFollowUp() []Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return takeQueued(&s.followUp, s.options.FollowUpMode)
}

func takeQueued(queue *[]Message, mode DeliveryMode) []Message {
	if len(*queue) == 0 {
		return nil
	}
	if mode == DeliveryAll {
		messages := cloneMessages(*queue)
		*queue = nil
		return messages
	}
	message := cloneMessage((*queue)[0])
	var zero Message
	(*queue)[0] = zero
	*queue = (*queue)[1:]
	return []Message{message}
}

func saveSessionSnapshot(ctx context.Context, store SessionStore, id string, snapshot SessionSnapshot) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("session store panicked: %v", recovered)
		}
	}()
	return store.SaveSession(ctx, id, snapshot)
}
