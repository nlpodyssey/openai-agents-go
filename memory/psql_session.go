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
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/jackc/pgx/v5"
)

// PgRowsInterface abstracts the rows operations for easier mocking
type PgRowsInterface interface {
	Next() bool
	Scan(dest ...any) error
	Err() error
	Close()
}

// PgRowInterface abstracts the row operations for easier mocking
type PgRowInterface interface {
	Scan(dest ...any) error
}

// PgConnInterface abstracts the database operations needed by PgSession.
// This allows for easy mocking in tests.
type PgConnInterface interface {
	Query(ctx context.Context, sql string, args ...any) (PgRowsInterface, error)
	QueryRow(ctx context.Context, sql string, args ...any) PgRowInterface
	Exec(ctx context.Context, sql string, args ...any) (any, error)
	Close(ctx context.Context) error
}

// PgRowsWrapper wraps pgx.Rows to implement PgRowsInterface
type PgRowsWrapper struct {
	rows pgx.Rows
}

func (w *PgRowsWrapper) Next() bool {
	return w.rows.Next()
}

func (w *PgRowsWrapper) Scan(dest ...any) error {
	return w.rows.Scan(dest...)
}

func (w *PgRowsWrapper) Err() error {
	return w.rows.Err()
}

func (w *PgRowsWrapper) Close() {
	w.rows.Close()
}

// PgRowWrapper wraps pgx.Row to implement PgRowInterface
type PgRowWrapper struct {
	row pgx.Row
}

func (w *PgRowWrapper) Scan(dest ...any) error {
	return w.row.Scan(dest...)
}

// PgConnWrapper wraps a real pgx.Conn to implement PgConnInterface
type PgConnWrapper struct {
	conn *pgx.Conn
}

func (w *PgConnWrapper) Query(ctx context.Context, sql string, args ...any) (PgRowsInterface, error) {
	rows, err := w.conn.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	return &PgRowsWrapper{rows: rows}, nil
}

func (w *PgConnWrapper) QueryRow(ctx context.Context, sql string, args ...any) PgRowInterface {
	row := w.conn.QueryRow(ctx, sql, args...)
	return &PgRowWrapper{row: row}
}

func (w *PgConnWrapper) Exec(ctx context.Context, sql string, args ...any) (any, error) {
	return w.conn.Exec(ctx, sql, args...)
}

func (w *PgConnWrapper) Close(ctx context.Context) error {
	return w.conn.Close(ctx)
}

// PgSession is a PostgreSQL-based implementation of Session storage.
//
// This implementation stores conversation history in a PostgreSQL database.
// Requires a valid PostgreSQL connection string.
type PgSession struct {
	sessionID     string
	connString    string
	sessionTable  string
	messagesTable string
	conn          PgConnInterface
	mu            sync.Mutex
}

type PgSessionParams struct {
	// Unique identifier for the conversation session
	SessionID string

	// PostgreSQL connection string.
	// Example: "postgres://user:password@localhost:5432/database"
	ConnectionString string

	// Optional name of the table to store session metadata.
	// Defaults to "agent_sessions".
	SessionTable string

	// Optional name of the table to store message data.
	// Defaults to "agent_messages".
	MessagesTable string

	// Optional connection interface for dependency injection (mainly for testing)
	Conn PgConnInterface
}

// NewPgSession initializes the PostgreSQL session.
func NewPgSession(ctx context.Context, params PgSessionParams) (_ *PgSession, err error) {
	s := &PgSession{
		sessionID:     params.SessionID,
		connString:    params.ConnectionString,
		sessionTable:  cmp.Or(params.SessionTable, "agent_sessions"),
		messagesTable: cmp.Or(params.MessagesTable, "agent_messages"),
		conn:          params.Conn,
	}

	defer func() {
		if err != nil {
			if s.conn != nil {
				if e := s.conn.Close(ctx); e != nil {
					err = errors.Join(err, e)
				}
			}
		}
	}()

	// If no connection provided, create a real one
	if s.conn == nil {
		if params.ConnectionString == "" {
			return nil, fmt.Errorf("connection string is required")
		}

		realConn, err := pgx.Connect(ctx, s.connString)
		if err != nil {
			return nil, fmt.Errorf("failed to connect to PostgreSQL: %w", err)
		}
		s.conn = &PgConnWrapper{conn: realConn}
	}

	err = s.initDB(ctx)
	if err != nil {
		return nil, err
	}
	return s, nil
}

func (s *PgSession) SessionID(context.Context) string {
	return s.sessionID
}

func (s *PgSession) GetItems(ctx context.Context, limit int) (_ []TResponseInputItem, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var rows PgRowsInterface
	if limit <= 0 {
		// Fetch all items in chronological order
		rows, err = s.conn.Query(ctx, fmt.Sprintf(`
			SELECT message_data FROM %s
			WHERE session_id = $1
			ORDER BY created_at ASC
		`, s.messagesTable), s.sessionID)
	} else {
		// Fetch the latest N items in chronological order
		rows, err = s.conn.Query(ctx, fmt.Sprintf(`
			SELECT message_data FROM %s
			WHERE session_id = $1
			ORDER BY created_at DESC
			LIMIT $2
		`, s.messagesTable), s.sessionID, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("error querying session items: %w", err)
	}
	defer rows.Close()

	var items []TResponseInputItem
	for rows.Next() {
		var messageData string
		if err = rows.Scan(&messageData); err != nil {
			return nil, fmt.Errorf("pgx rows scan error: %w", err)
		}

		item, err := unmarshalMessageData(messageData)
		if err != nil {
			continue // Skip invalid JSON entries
		}
		items = append(items, item)
	}
	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("pgx rows scan error: %w", err)
	}

	// Reverse to get chronological order when using DESC
	if limit > 0 {
		slices.Reverse(items)
	}

	return items, nil
}

func (s *PgSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	if len(items) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Ensure session exists
	_, err := s.conn.Exec(
		ctx,
		fmt.Sprintf(`INSERT INTO %s (session_id) VALUES ($1) ON CONFLICT (session_id) DO NOTHING`, s.sessionTable),
		s.sessionID,
	)
	if err != nil {
		return fmt.Errorf("error ensuring session exists: %w", err)
	}

	// Add items
	for _, item := range items {
		jsonItem, err := item.MarshalJSON()
		if err != nil {
			return fmt.Errorf("error JSON marshaling item: %w", err)
		}
		_, err = s.conn.Exec(
			ctx,
			fmt.Sprintf(`INSERT INTO %s (session_id, message_data) VALUES ($1, $2)`, s.messagesTable),
			s.sessionID, string(jsonItem),
		)
		if err != nil {
			return fmt.Errorf("error inserting item in messages table: %w", err)
		}
	}

	// Update session timestamp
	_, err = s.conn.Exec(
		ctx,
		fmt.Sprintf(`UPDATE %s SET updated_at = NOW() WHERE session_id = $1`, s.sessionTable),
		s.sessionID,
	)
	if err != nil {
		return fmt.Errorf("error updating session timestamp: %w", err)
	}

	return nil
}

func (s *PgSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var messageData string
	err := s.conn.QueryRow(
		ctx,
		fmt.Sprintf(`
			DELETE FROM %s
			WHERE id = (
				SELECT id FROM %s
				WHERE session_id = $1
				ORDER BY created_at DESC
				LIMIT 1
			)
			RETURNING message_data
		`, s.messagesTable, s.messagesTable),
		s.sessionID,
	).Scan(&messageData)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("error popping item: %w", err)
	}

	item, err := unmarshalMessageData(messageData)
	if err != nil {
		return nil, nil // Return nil for corrupted JSON entries (already deleted)
	}

	return &item, nil
}

func (s *PgSession) ClearSession(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.conn.Exec(
		ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE session_id = $1`, s.messagesTable),
		s.sessionID,
	)
	if err != nil {
		return fmt.Errorf("error clearing messages: %w", err)
	}

	_, err = s.conn.Exec(
		ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE session_id = $1`, s.sessionTable),
		s.sessionID,
	)
	if err != nil {
		return fmt.Errorf("error clearing session: %w", err)
	}

	return nil
}

// Initialize the database schema.
func (s *PgSession) initDB(ctx context.Context) error {
	_, err := s.conn.Exec(ctx, fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			session_id TEXT PRIMARY KEY,
			created_at TIMESTAMP DEFAULT NOW(),
			updated_at TIMESTAMP DEFAULT NOW()
		)
	`, s.sessionTable))
	if err != nil {
		return fmt.Errorf("error creating session table: %w", err)
	}

	_, err = s.conn.Exec(ctx, fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id SERIAL PRIMARY KEY,
			session_id TEXT NOT NULL,
			message_data TEXT NOT NULL,
			created_at TIMESTAMP DEFAULT NOW(),
			FOREIGN KEY (session_id) REFERENCES %s (session_id) ON DELETE CASCADE
		)
	`, s.messagesTable, s.sessionTable))
	if err != nil {
		return fmt.Errorf("error creating messages table: %w", err)
	}

	_, err = s.conn.Exec(ctx, fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS idx_%s_session_id ON %s (session_id, created_at)`,
		s.messagesTable, s.messagesTable))
	if err != nil {
		return fmt.Errorf("error creating index: %w", err)
	}

	return nil
}

// Close the database connection.
func (s *PgSession) Close(ctx context.Context) error {
	if s.conn != nil {
		return s.conn.Close(ctx)
	}
	return nil
}
