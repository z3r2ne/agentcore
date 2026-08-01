package agentcore

import (
	"context"
	"fmt"
	"sync"
)

// EventStream is an asynchronous, unbounded event iterator for one agent run.
// Call Next until it returns false, then call Result. Events are queued in
// memory so a temporarily slow consumer never blocks model or tool execution.
type EventStream struct {
	mu     sync.Mutex
	ready  *sync.Cond
	events []Event
	closed bool
	result Result
	err    error
}

// Stream starts Prompt in a goroutine and returns its event iterator.
func (a *Agent) Stream(ctx context.Context, state State, prompts []Message) *EventStream {
	return newEventStream(func(sink EventSink) (Result, error) {
		return a.Prompt(ctx, state, prompts, sink)
	})
}

func newEventStream(run func(EventSink) (Result, error)) *EventStream {
	stream := &EventStream{}
	stream.ready = sync.NewCond(&stream.mu)
	go func() {
		var result Result
		var err error
		defer func() {
			if recovered := recover(); recovered != nil {
				err = fmt.Errorf("agentcore: asynchronous run panicked: %v", recovered)
			}
			stream.mu.Lock()
			stream.result = result
			stream.err = err
			stream.closed = true
			stream.ready.Broadcast()
			stream.mu.Unlock()
		}()
		result, err = run(func(_ context.Context, event Event) error {
			stream.mu.Lock()
			stream.events = append(stream.events, event)
			stream.ready.Signal()
			stream.mu.Unlock()
			return nil
		})
	}()
	return stream
}

// Next returns the next event. It blocks while the run is active and no event
// is currently available, and returns false after all events are consumed.
func (s *EventStream) Next() (Event, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.events) == 0 && !s.closed {
		s.ready.Wait()
	}
	if len(s.events) == 0 {
		return Event{}, false
	}
	event := s.events[0]
	var zero Event
	s.events[0] = zero
	s.events = s.events[1:]
	return event, true
}

// Result waits for execution to finish and returns its result. It does not
// require events to be drained first.
func (s *EventStream) Result() (Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for !s.closed {
		s.ready.Wait()
	}
	return s.result, s.err
}
