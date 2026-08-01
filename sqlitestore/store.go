package sqlitestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/z3r2ne/agentcore"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"gorm.io/gorm/logger"
)

const schemaVersion = 1

var memoryDatabaseSequence atomic.Uint64

// ErrSessionNotFound indicates that a requested session ID does not exist.
var ErrSessionNotFound = errors.New("agentcore/sqlitestore: session not found")

// Store owns or wraps a migrated SQLite database.
type Store struct {
	db    *gorm.DB
	owned bool
}

type sessionRecord struct {
	ID            string    `gorm:"primaryKey;size:255"`
	SchemaVersion int       `gorm:"not null"`
	Payload       string    `gorm:"type:text;not null"`
	MessageCount  int       `gorm:"not null"`
	CreatedAt     time.Time `gorm:"not null"`
	UpdatedAt     time.Time `gorm:"not null;index"`
}

type sessionPayload struct {
	Snapshot         agentcore.SessionSnapshot `json:"snapshot"`
	RawToolArguments []rawToolArguments        `json:"rawToolArguments,omitempty"`
	RawProviderData  []rawProviderData         `json:"rawProviderData,omitempty"`
}

type rawToolArguments struct {
	Collection string `json:"collection"`
	Message    int    `json:"message"`
	Block      int    `json:"block"`
	Data       []byte `json:"data"`
}

type rawProviderData struct {
	Collection string `json:"collection"`
	Message    int    `json:"message"`
	Data       []byte `json:"data"`
}

func (sessionRecord) TableName() string { return "agentcore_sessions" }

// SessionInfo is lightweight metadata returned by ListSessions.
type SessionInfo struct {
	ID            string    `json:"id"`
	MessageCount  int       `json:"messageCount"`
	SchemaVersion int       `json:"schemaVersion"`
	CreatedAt     time.Time `json:"createdAt"`
	UpdatedAt     time.Time `json:"updatedAt"`
}

// Open creates or opens a SQLite file, enables WAL and a busy timeout, and
// initializes the agentcore_sessions table.
func Open(path string) (*Store, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, errors.New("agentcore/sqlitestore: database path is required")
	}
	dsn := path
	filePath := ""
	if path == ":memory:" {
		dsn = fmt.Sprintf("file:agentcore-memory-%d?mode=memory&cache=shared&_foreign_keys=on", memoryDatabaseSequence.Add(1))
	} else if !strings.HasPrefix(path, "file:") {
		absolute, err := filepath.Abs(path)
		if err != nil {
			return nil, fmt.Errorf("agentcore/sqlitestore: resolve database path: %w", err)
		}
		if err := os.MkdirAll(filepath.Dir(absolute), 0o700); err != nil {
			return nil, fmt.Errorf("agentcore/sqlitestore: create database directory: %w", err)
		}
		filePath = absolute
		dsn = absolute + "?_journal_mode=WAL&_busy_timeout=30000&_foreign_keys=on&_synchronous=NORMAL"
	}
	db, err := gorm.Open(sqlite.Open(dsn), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		return nil, fmt.Errorf("agentcore/sqlitestore: open SQLite: %w", err)
	}
	if err := configureSQLitePool(db); err != nil {
		if sqlDB, dbErr := db.DB(); dbErr == nil {
			_ = sqlDB.Close()
		}
		return nil, err
	}
	store, err := New(db)
	if err != nil {
		if sqlDB, dbErr := db.DB(); dbErr == nil {
			_ = sqlDB.Close()
		}
		return nil, err
	}
	store.owned = true
	if filePath != "" {
		if err := os.Chmod(filePath, 0o600); err != nil {
			_ = store.Close()
			return nil, fmt.Errorf("agentcore/sqlitestore: secure database file: %w", err)
		}
	}
	return store, nil
}

// New wraps an existing GORM connection and initializes the session table.
// Close does not close connections supplied through New.
func New(db *gorm.DB) (*Store, error) {
	if db == nil {
		return nil, errors.New("agentcore/sqlitestore: nil database")
	}
	if db.Dialector == nil || db.Dialector.Name() != "sqlite" {
		return nil, errors.New("agentcore/sqlitestore: database is not SQLite")
	}
	if err := db.AutoMigrate(&sessionRecord{}); err != nil {
		return nil, fmt.Errorf("agentcore/sqlitestore: initialize schema: %w", err)
	}
	return &Store{db: db}, nil
}

// SaveSession atomically inserts or replaces one durable session snapshot.
func (s *Store) SaveSession(ctx context.Context, id string, snapshot agentcore.SessionSnapshot) error {
	if s == nil || s.db == nil {
		return errors.New("agentcore/sqlitestore: nil store")
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return errors.New("agentcore/sqlitestore: session ID is required")
	}
	payload, err := encodeSnapshot(snapshot)
	if err != nil {
		return fmt.Errorf("agentcore/sqlitestore: encode session %q: %w", id, err)
	}
	now := time.Now().UTC()
	record := sessionRecord{
		ID: id, SchemaVersion: schemaVersion, Payload: string(payload),
		MessageCount: len(snapshot.State.Messages), CreatedAt: now, UpdatedAt: now,
	}
	err = s.db.WithContext(nonNilContext(ctx)).Clauses(clause.OnConflict{
		Columns: []clause.Column{{Name: "id"}},
		DoUpdates: clause.Assignments(map[string]any{
			"schema_version": schemaVersion,
			"payload":        record.Payload,
			"message_count":  record.MessageCount,
			"updated_at":     now,
		}),
	}).Create(&record).Error
	if err != nil {
		return fmt.Errorf("agentcore/sqlitestore: save session %q: %w", id, err)
	}
	return nil
}

// LoadSession returns a detached durable snapshot.
func (s *Store) LoadSession(ctx context.Context, id string) (agentcore.SessionSnapshot, error) {
	if s == nil || s.db == nil {
		return agentcore.SessionSnapshot{}, errors.New("agentcore/sqlitestore: nil store")
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return agentcore.SessionSnapshot{}, errors.New("agentcore/sqlitestore: session ID is required")
	}
	var record sessionRecord
	err := s.db.WithContext(nonNilContext(ctx)).First(&record, "id = ?", id).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return agentcore.SessionSnapshot{}, ErrSessionNotFound
	}
	if err != nil {
		return agentcore.SessionSnapshot{}, fmt.Errorf("agentcore/sqlitestore: load session %q: %w", id, err)
	}
	if record.SchemaVersion != schemaVersion {
		return agentcore.SessionSnapshot{}, fmt.Errorf("agentcore/sqlitestore: session %q uses unsupported schema version %d", id, record.SchemaVersion)
	}
	snapshot, err := decodeSnapshot([]byte(record.Payload))
	if err != nil {
		return agentcore.SessionSnapshot{}, fmt.Errorf("agentcore/sqlitestore: decode session %q: %w", id, err)
	}
	return snapshot, nil
}

// RestoreSession loads a snapshot and attaches automatic future checkpoints to
// the same Store and ID.
func (s *Store) RestoreSession(ctx context.Context, id string, agent *agentcore.Agent, options agentcore.SessionOptions) (*agentcore.Session, error) {
	snapshot, err := s.LoadSession(ctx, id)
	if err != nil {
		return nil, err
	}
	options.Store = s
	options.SessionID = id
	return agentcore.NewSessionFromSnapshot(agent, snapshot, options)
}

// DeleteSession removes one session. Missing IDs return ErrSessionNotFound.
func (s *Store) DeleteSession(ctx context.Context, id string) error {
	if s == nil || s.db == nil {
		return errors.New("agentcore/sqlitestore: nil store")
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return errors.New("agentcore/sqlitestore: session ID is required")
	}
	result := s.db.WithContext(nonNilContext(ctx)).Delete(&sessionRecord{}, "id = ?", id)
	if result.Error != nil {
		return fmt.Errorf("agentcore/sqlitestore: delete session %q: %w", id, result.Error)
	}
	if result.RowsAffected == 0 {
		return ErrSessionNotFound
	}
	return nil
}

// ListSessions returns sessions ordered by most recently updated first.
// A non-positive limit defaults to 100.
func (s *Store) ListSessions(ctx context.Context, limit, offset int) ([]SessionInfo, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("agentcore/sqlitestore: nil store")
	}
	if limit <= 0 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}
	var records []sessionRecord
	err := s.db.WithContext(nonNilContext(ctx)).Select("id", "message_count", "schema_version", "created_at", "updated_at").
		Order("updated_at DESC, id ASC").Limit(limit).Offset(offset).Find(&records).Error
	if err != nil {
		return nil, fmt.Errorf("agentcore/sqlitestore: list sessions: %w", err)
	}
	result := make([]SessionInfo, len(records))
	for index, record := range records {
		result[index] = SessionInfo{
			ID: record.ID, MessageCount: record.MessageCount, SchemaVersion: record.SchemaVersion,
			CreatedAt: record.CreatedAt, UpdatedAt: record.UpdatedAt,
		}
	}
	return result, nil
}

// Close closes the database only when it was created by Open.
func (s *Store) Close() error {
	if s == nil || s.db == nil || !s.owned {
		return nil
	}
	sqlDB, err := s.db.DB()
	if err != nil {
		return err
	}
	s.owned = false
	return sqlDB.Close()
}

func nonNilContext(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

func configureSQLitePool(db *gorm.DB) error {
	sqlDB, err := db.DB()
	if err != nil {
		return fmt.Errorf("agentcore/sqlitestore: configure pool: %w", err)
	}
	// SQLite serializes writes. One connection avoids SQLITE_BUSY contention
	// within this process while WAL still permits readers from other processes.
	sqlDB.SetMaxOpenConns(1)
	sqlDB.SetMaxIdleConns(1)
	sqlDB.SetConnMaxLifetime(0)
	return nil
}

func encodeSnapshot(snapshot agentcore.SessionSnapshot) ([]byte, error) {
	payload := sessionPayload{Snapshot: snapshot}
	payload.Snapshot.State.Messages = sanitizeMessages(snapshot.State.Messages, "state", &payload)
	payload.Snapshot.Steering = sanitizeMessages(snapshot.Steering, "steering", &payload)
	payload.Snapshot.FollowUp = sanitizeMessages(snapshot.FollowUp, "follow_up", &payload)
	return json.Marshal(payload)
}

func sanitizeMessages(messages []agentcore.Message, collection string, payload *sessionPayload) []agentcore.Message {
	result := append([]agentcore.Message(nil), messages...)
	for messageIndex := range result {
		message := &result[messageIndex]
		message.Content = append([]agentcore.ContentBlock(nil), message.Content...)
		if message.ProviderData != nil {
			providerData := *message.ProviderData
			if !json.Valid(providerData.Data) {
				payload.RawProviderData = append(payload.RawProviderData, rawProviderData{
					Collection: collection, Message: messageIndex, Data: append([]byte(nil), providerData.Data...),
				})
				providerData.Data = json.RawMessage("null")
			}
			message.ProviderData = &providerData
		}
		for blockIndex := range message.Content {
			block := &message.Content[blockIndex]
			if block.ToolCall == nil || json.Valid(block.ToolCall.Arguments) {
				continue
			}
			call := *block.ToolCall
			payload.RawToolArguments = append(payload.RawToolArguments, rawToolArguments{
				Collection: collection, Message: messageIndex, Block: blockIndex, Data: append([]byte(nil), call.Arguments...),
			})
			call.Arguments = json.RawMessage("null")
			block.ToolCall = &call
		}
	}
	return result
}

func decodeSnapshot(data []byte) (agentcore.SessionSnapshot, error) {
	var payload sessionPayload
	if err := json.Unmarshal(data, &payload); err != nil {
		return agentcore.SessionSnapshot{}, err
	}
	for _, raw := range payload.RawToolArguments {
		messages, err := snapshotMessages(&payload.Snapshot, raw.Collection)
		if err != nil {
			return agentcore.SessionSnapshot{}, err
		}
		if raw.Message < 0 || raw.Message >= len(*messages) {
			return agentcore.SessionSnapshot{}, errors.New("invalid raw tool argument message index")
		}
		message := &(*messages)[raw.Message]
		if raw.Block < 0 || raw.Block >= len(message.Content) || message.Content[raw.Block].ToolCall == nil {
			return agentcore.SessionSnapshot{}, errors.New("invalid raw tool argument block index")
		}
		message.Content[raw.Block].ToolCall.Arguments = append(json.RawMessage(nil), raw.Data...)
	}
	for _, raw := range payload.RawProviderData {
		messages, err := snapshotMessages(&payload.Snapshot, raw.Collection)
		if err != nil {
			return agentcore.SessionSnapshot{}, err
		}
		if raw.Message < 0 || raw.Message >= len(*messages) || (*messages)[raw.Message].ProviderData == nil {
			return agentcore.SessionSnapshot{}, errors.New("invalid raw provider data message index")
		}
		(*messages)[raw.Message].ProviderData.Data = append(json.RawMessage(nil), raw.Data...)
	}
	return payload.Snapshot, nil
}

func snapshotMessages(snapshot *agentcore.SessionSnapshot, collection string) (*[]agentcore.Message, error) {
	switch collection {
	case "state":
		return &snapshot.State.Messages, nil
	case "steering":
		return &snapshot.Steering, nil
	case "follow_up":
		return &snapshot.FollowUp, nil
	default:
		return nil, fmt.Errorf("invalid message collection %q", collection)
	}
}
