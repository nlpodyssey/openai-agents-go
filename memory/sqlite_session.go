// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package memory

import (
	"cmp"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/openai/openai-go/v3/shared/constant"

	_ "github.com/mattn/go-sqlite3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

// SQLiteSession is a SQLite-based implementation of Session storage.
//
// This implementation stores conversation history in a SQLite database.
// By default, uses an in-memory database that is lost when the process ends.
// For persistent storage, provide a file path.
type SQLiteSession struct {
	sessionID     string
	dbDSN         string
	sessionTable  string
	messagesTable string
	db            *sql.DB
	mu            sync.Mutex
}

type SQLiteSessionParams struct {
	// Unique identifier for the conversation session
	SessionID string

	// Optional database data source name.
	// Defaults to ':memory:' (in-memory database).
	DBDataSourceName string

	// Optional name of the table to store session metadata.
	// Defaults to "file::memory:?cache=shared".
	SessionTable string

	// Optional name of the table to store message data.
	// Defaults to "agent_messages".
	MessagesTable string
}

// NewSQLiteSession initializes the SQLite session.
func NewSQLiteSession(ctx context.Context, params SQLiteSessionParams) (_ *SQLiteSession, err error) {
	s := &SQLiteSession{
		sessionID:     params.SessionID,
		dbDSN:         cmp.Or(params.DBDataSourceName, "file::memory:?cache=shared"),
		sessionTable:  cmp.Or(params.SessionTable, "agent_sessions"),
		messagesTable: cmp.Or(params.MessagesTable, "agent_messages"),
	}

	defer func() {
		if err != nil {
			if e := s.Close(); e != nil {
				err = errors.Join(err, e)
			}
		}
	}()

	s.db, err = sql.Open("sqlite3", s.dbDSN)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite3 database: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `PRAGMA journal_mode=WAL`)
	if err != nil {
		return nil, fmt.Errorf("failed to set journal mode: %w", err)
	}

	err = s.initDB(ctx)
	if err != nil {
		return nil, err
	}
	return s, nil
}

func (s *SQLiteSession) SessionID(context.Context) string {
	return s.sessionID
}

func (s *SQLiteSession) GetItems(ctx context.Context, limit int) (_ []TResponseInputItem, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var rows *sql.Rows
	if limit <= 0 {
		// Fetch all items in chronological order
		rows, err = s.db.QueryContext(ctx, fmt.Sprintf(`
			SELECT message_data FROM "%s"
			WHERE session_id = ?
			ORDER BY created_at ASC
		`, s.messagesTable), s.sessionID)
	} else {
		// Fetch the latest N items in chronological order
		rows, err = s.db.QueryContext(ctx, fmt.Sprintf(`
			SELECT message_data FROM "%s"
			WHERE session_id = ?
			ORDER BY created_at DESC
			LIMIT ?
		`, s.messagesTable), s.sessionID, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("error querying session items: %w", err)
	}
	defer func() {
		if e := rows.Close(); e != nil {
			err = errors.Join(err, fmt.Errorf("error closing sql.Rows: %w", e))
		}
	}()

	var items []TResponseInputItem
	for rows.Next() {
		var messageData string
		if err = rows.Scan(&messageData); err != nil {
			return nil, fmt.Errorf("sql rows scan error: %w", err)
		}

		item, err := unmarshalMessageData(messageData)
		if err != nil {
			continue // Skip invalid JSON entries
		}
		items = append(items, item)
	}
	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("sql rows scan error: %w", err)
	}

	// Reverse to get chronological order when using DESC
	if limit > 0 {
		slices.Reverse(items)
	}

	// xxx call and xxx call output must appear in pairs
	// so remove the first item if it's a xxx call output
	if len(items) > 0 {
		switch *items[0].GetType() {
		case string(constant.ValueOf[constant.FunctionCallOutput]()):
			items = slices.Delete(items, 0, 1)
		case string(constant.ValueOf[constant.ComputerCallOutput]()):
			items = slices.Delete(items, 0, 1)
		case string(constant.ValueOf[constant.LocalShellCallOutput]()):
			items = slices.Delete(items, 0, 1)
		case string(constant.ValueOf[constant.CustomToolCallOutput]()):
			items = slices.Delete(items, 0, 1)
		}
	}

	return items, nil
}

func (s *SQLiteSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	if len(items) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Ensure session exists
	_, err := s.db.ExecContext(
		ctx,
		fmt.Sprintf(`INSERT OR IGNORE INTO "%s" (session_id) VALUES (?)`, s.sessionTable),
		s.sessionID,
	)
	if err != nil {
		return err
	}

	// Add items
	for _, item := range items {
		jsonItem, err := item.MarshalJSON()
		if err != nil {
			return fmt.Errorf("error JSON marshaling item: %w", err)
		}
		_, err = s.db.ExecContext(
			ctx,
			fmt.Sprintf(`INSERT INTO "%s" (session_id, message_data) VALUES (?, ?)`, s.messagesTable),
			s.sessionID, string(jsonItem),
		)
		if err != nil {
			return fmt.Errorf("error inserting item in messages table: %w", err)
		}
	}

	// Update session timestamp
	_, err = s.db.ExecContext(
		ctx,
		fmt.Sprintf(`UPDATE "%s" SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?`, s.sessionTable),
		s.sessionID,
	)
	if err != nil {
		return fmt.Errorf("error updating session timestamp: %w", err)
	}

	return nil
}

func (s *SQLiteSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	var messageData string
	err := s.db.QueryRowContext(
		ctx,
		// Use DELETE with RETURNING to atomically delete and return the most recent item
		fmt.Sprintf(`
			DELETE FROM "%s"
			WHERE id = (
				SELECT id FROM "%s"
				WHERE session_id = ?
				ORDER BY created_at DESC
				LIMIT 1
			)
			RETURNING message_data
		`, s.messagesTable, s.messagesTable),
		s.sessionID,
	).Scan(&messageData)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	item, err := unmarshalMessageData(messageData)
	if err != nil {
		return nil, nil // Return nil for corrupted JSON entries (already deleted)
	}

	return &item, nil
}

func (s *SQLiteSession) ClearSession(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.ExecContext(
		ctx,
		fmt.Sprintf(`DELETE FROM "%s" WHERE session_id = ?`, s.messagesTable),
		s.sessionID,
	)
	if err != nil {
		return err
	}

	_, err = s.db.ExecContext(
		ctx,
		fmt.Sprintf(`DELETE FROM "%s" WHERE session_id = ?`, s.sessionTable),
		s.sessionID,
	)
	if err != nil {
		return err
	}

	return nil
}

// Initialize the database schema.
func (s *SQLiteSession) initDB(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS "%s" (
			session_id TEXT PRIMARY KEY,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`, s.sessionTable))
	if err != nil {
		return fmt.Errorf("error creating session table: %w", err)
	}

	_, err = s.db.ExecContext(ctx, fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS "%s" (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL,
			message_data TEXT NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (session_id) REFERENCES "%s" (session_id) ON DELETE CASCADE
		)
	`, s.messagesTable, s.sessionTable))
	if err != nil {
		return fmt.Errorf("error creating messages table: %w", err)
	}

	_, err = s.db.ExecContext(ctx, fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS "idx_%s_session_id" ON "%s" (session_id, created_at)`,
		s.messagesTable, s.messagesTable))
	if err != nil {
		return fmt.Errorf("error creating index: %w", err)
	}

	return nil
}

// Close the database connection.
func (s *SQLiteSession) Close() error {
	return s.db.Close()
}

func unmarshalMessageData(messageData string) (TResponseInputItem, error) {
	var item TResponseInputItem
	err := json.Unmarshal([]byte(messageData), &item)
	if err != nil {
		return TResponseInputItem{}, err
	}

	// Fix incorrect output messages unmarshaling
	if msg := item.OfMessage; !param.IsOmitted(msg) {
		if msg.Content.OfInputItemContentList == nil && msg.Content.OfString == (param.Opt[string]{}) {
			var outMsg responses.ResponseOutputMessageParam
			err = json.Unmarshal([]byte(messageData), &outMsg)
			if err == nil && len(outMsg.Content) > 0 && !param.IsOmitted(outMsg.Content[0].OfOutputText) && outMsg.Content[0].OfOutputText.Text != "" {
				item = TResponseInputItem{
					OfOutputMessage: &outMsg,
				}
			}
		}
	}

	return item, nil
}
