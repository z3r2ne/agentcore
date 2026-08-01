package sqlitestore_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/z3r2ne/agentcore"
	"github.com/z3r2ne/agentcore/sqlitestore"
)

type queuedModel struct {
	mu        sync.Mutex
	responses [][]agentcore.ModelChunk
}

func (m *queuedModel) Stream(context.Context, agentcore.ModelRequest) (agentcore.ModelStream, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.responses) == 0 {
		return nil, errors.New("no response")
	}
	chunks := append([]agentcore.ModelChunk(nil), m.responses[0]...)
	m.responses = m.responses[1:]
	return &chunkStream{chunks: chunks}, nil
}

type chunkStream struct {
	chunks []agentcore.ModelChunk
}

func (s *chunkStream) Recv() (agentcore.ModelChunk, error) {
	if len(s.chunks) == 0 {
		return agentcore.ModelChunk{}, io.EOF
	}
	chunk := s.chunks[0]
	s.chunks = s.chunks[1:]
	return chunk, nil
}

func (*chunkStream) Close() error { return nil }

type blockingModel struct {
	started chan struct{}
}

func (m *blockingModel) Stream(ctx context.Context, _ agentcore.ModelRequest) (agentcore.ModelStream, error) {
	close(m.started)
	return &blockingStream{ctx: ctx}, nil
}

type blockingStream struct {
	ctx context.Context
}

func (s *blockingStream) Recv() (agentcore.ModelChunk, error) {
	<-s.ctx.Done()
	return agentcore.ModelChunk{}, s.ctx.Err()
}

func (*blockingStream) Close() error { return nil }

func openTestStore(t *testing.T) *sqlitestore.Store {
	t.Helper()
	store, err := sqlitestore.Open(filepath.Join(t.TempDir(), "sessions.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close store: %v", err)
		}
	})
	return store
}

func TestSaveLoadListAndDeleteSession(t *testing.T) {
	store := openTestStore(t)
	snapshot := agentcore.SessionSnapshot{
		State: agentcore.State{Messages: []agentcore.Message{{
			Role: agentcore.RoleAssistant,
			Content: []agentcore.ContentBlock{
				{Type: agentcore.ContentToolCall, ToolCall: &agentcore.ToolCall{ID: "call-1", Name: "broken", Arguments: json.RawMessage(`{"unfinished"`)}},
				{Type: agentcore.ContentImage, Data: []byte("image"), MIMEType: "image/png"},
			},
			ProviderData: &agentcore.ProviderData{Format: "opaque/v1", Data: json.RawMessage("not-json"), Runtime: func() {}},
		}}},
		Usage:        agentcore.Usage{InputTokens: 10, OutputTokens: 4},
		Steering:     []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "steer")},
		FollowUp:     []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "follow")},
		SteeringMode: agentcore.DeliveryAll, FollowUpMode: agentcore.DeliveryOne,
	}
	if err := store.SaveSession(context.Background(), "session-1", snapshot); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.LoadSession(context.Background(), "session-1")
	if err != nil {
		t.Fatal(err)
	}
	if got := string(loaded.State.Messages[0].ToolCalls()[0].Arguments); got != `{"unfinished"` {
		t.Fatalf("tool arguments = %q", got)
	}
	if got := string(loaded.State.Messages[0].ProviderData.Data); got != "not-json" || loaded.State.Messages[0].ProviderData.Runtime != nil {
		t.Fatalf("provider data = %+v", loaded.State.Messages[0].ProviderData)
	}
	if !reflect.DeepEqual(loaded.Usage, snapshot.Usage) || loaded.SteeringMode != agentcore.DeliveryAll || loaded.FollowUp[0].Text() != "follow" || string(loaded.State.Messages[0].Content[1].Data) != "image" {
		t.Fatalf("loaded = %+v", loaded)
	}
	infos, err := store.ListSessions(context.Background(), 10, 0)
	if err != nil || len(infos) != 1 || infos[0].ID != "session-1" || infos[0].MessageCount != 1 {
		t.Fatalf("infos=%+v err=%v", infos, err)
	}
	if err := store.DeleteSession(context.Background(), "session-1"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.LoadSession(context.Background(), "session-1"); !errors.Is(err, sqlitestore.ErrSessionNotFound) {
		t.Fatalf("load after delete = %v", err)
	}
}

func TestSessionAutomaticallyCheckpointsAndRestores(t *testing.T) {
	store := openTestStore(t)
	model := &queuedModel{responses: [][]agentcore.ModelChunk{{{
		TextDelta: "first", StopReason: agentcore.StopReasonStop,
		Usage: &agentcore.Usage{InputTokens: 3, OutputTokens: 2},
	}}}}
	agent, err := agentcore.New(agentcore.Config{Model: model})
	if err != nil {
		t.Fatal(err)
	}
	session, err := agentcore.NewSession(agent, agentcore.State{}, agentcore.SessionOptions{
		Store: store, SessionID: "durable", SteeringMode: agentcore.DeliveryAll,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Prompt(context.Background(), []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "hello")}, nil); err != nil {
		t.Fatal(err)
	}
	snapshot, err := store.LoadSession(context.Background(), "durable")
	if err != nil || len(snapshot.State.Messages) != 2 || snapshot.State.Messages[1].Text() != "first" || snapshot.Usage.OutputTokens != 2 {
		t.Fatalf("snapshot=%+v err=%v", snapshot, err)
	}

	restoredModel := &queuedModel{responses: [][]agentcore.ModelChunk{{{TextDelta: "second", StopReason: agentcore.StopReasonStop}}}}
	restoredAgent, _ := agentcore.New(agentcore.Config{Model: restoredModel})
	restored, err := store.RestoreSession(context.Background(), "durable", restoredAgent, agentcore.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if restored.Status().Usage.OutputTokens != 2 {
		t.Fatalf("restored status = %+v", restored.Status())
	}
	if _, err := restored.Prompt(context.Background(), []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "again")}, nil); err != nil {
		t.Fatal(err)
	}
	updated, err := store.LoadSession(context.Background(), "durable")
	if err != nil || len(updated.State.Messages) != 4 || updated.State.Messages[3].Text() != "second" {
		t.Fatalf("updated=%+v err=%v", updated, err)
	}
}

func TestActiveQueuesAreCheckpointed(t *testing.T) {
	store := openTestStore(t)
	model := &blockingModel{started: make(chan struct{})}
	agent, _ := agentcore.New(agentcore.Config{Model: model})
	session, err := agentcore.NewSession(agent, agentcore.State{}, agentcore.SessionOptions{Store: store, SessionID: "queued"})
	if err != nil {
		t.Fatal(err)
	}
	run := session.Stream(context.Background(), []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "start")})
	<-model.started
	if err := session.Steer(agentcore.TextMessage(agentcore.RoleUser, "steer")); err != nil {
		t.Fatal(err)
	}
	if err := session.FollowUp(agentcore.TextMessage(agentcore.RoleUser, "follow")); err != nil {
		t.Fatal(err)
	}
	checkpoint, err := store.LoadSession(context.Background(), "queued")
	if err != nil || len(checkpoint.Steering) != 1 || len(checkpoint.FollowUp) != 1 {
		t.Fatalf("checkpoint=%+v err=%v", checkpoint, err)
	}
	if err := session.Abort(); err != nil {
		t.Fatal(err)
	}
	if _, err := run.Result(); !errors.Is(err, context.Canceled) {
		t.Fatalf("run error = %v", err)
	}
	final, err := store.LoadSession(context.Background(), "queued")
	if err != nil || len(final.Steering) != 1 || len(final.FollowUp) != 1 {
		t.Fatalf("final=%+v err=%v", final, err)
	}
}

func TestConcurrentSavesRemainAtomic(t *testing.T) {
	store := openTestStore(t)
	var wait sync.WaitGroup
	errorsSeen := make(chan error, 32)
	for index := range 32 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			snapshot := agentcore.SessionSnapshot{State: agentcore.State{Messages: []agentcore.Message{
				agentcore.TextMessage(agentcore.RoleUser, string(rune('a'+index%26))),
			}}}
			if err := store.SaveSession(context.Background(), "shared", snapshot); err != nil {
				errorsSeen <- err
			}
		}()
	}
	wait.Wait()
	close(errorsSeen)
	for err := range errorsSeen {
		t.Fatal(err)
	}
	loaded, err := store.LoadSession(context.Background(), "shared")
	if err != nil || len(loaded.State.Messages) != 1 || loaded.State.Messages[0].Text() == "" {
		t.Fatalf("loaded=%+v err=%v", loaded, err)
	}
}
