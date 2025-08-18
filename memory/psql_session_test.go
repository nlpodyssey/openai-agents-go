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
	"context"
	"fmt"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/responses"
	"github.com/openai/openai-go/v2/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockPgConn is a mock implementation of PgConnInterface for testing
type MockPgConn struct {
	mock.Mock
}

func (m *MockPgConn) Query(ctx context.Context, sql string, args ...any) (PgRowsInterface, error) {
	arguments := []any{ctx, sql}
	arguments = append(arguments, args...)
	ret := m.Called(arguments...)
	return ret.Get(0).(PgRowsInterface), ret.Error(1)
}

func (m *MockPgConn) QueryRow(ctx context.Context, sql string, args ...any) PgRowInterface {
	arguments := []any{ctx, sql}
	arguments = append(arguments, args...)
	ret := m.Called(arguments...)
	return ret.Get(0).(PgRowInterface)
}

func (m *MockPgConn) Exec(ctx context.Context, sql string, args ...any) (any, error) {
	arguments := []any{ctx, sql}
	arguments = append(arguments, args...)
	ret := m.Called(arguments...)
	return ret.Get(0), ret.Error(1)
}

func (m *MockPgConn) Close(ctx context.Context) error {
	ret := m.Called(ctx)
	return ret.Error(0)
}

// MockPgRows is a mock implementation of PgRowsInterface for testing
type MockPgRows struct {
	data []string
	pos  int
}

func NewMockPgRows(data []string) *MockPgRows {
	return &MockPgRows{data: data, pos: -1}
}

func (m *MockPgRows) Next() bool {
	m.pos++
	return m.pos < len(m.data)
}

func (m *MockPgRows) Scan(dest ...any) error {
	if m.pos >= len(m.data) {
		return fmt.Errorf("no more rows")
	}
	if len(dest) > 0 {
		if strPtr, ok := dest[0].(*string); ok {
			*strPtr = m.data[m.pos]
		}
	}
	return nil
}

func (m *MockPgRows) Err() error {
	return nil
}

func (m *MockPgRows) Close() {}

// MockPgRow is a mock implementation of PgRowInterface for testing
type MockPgRow struct {
	data  string
	empty bool
}

func NewMockPgRow(data string, empty bool) *MockPgRow {
	return &MockPgRow{data: data, empty: empty}
}

func (m *MockPgRow) Scan(dest ...any) error {
	if m.empty {
		return pgx.ErrNoRows
	}
	if len(dest) > 0 {
		if strPtr, ok := dest[0].(*string); ok {
			*strPtr = m.data
		}
	}
	return nil
}

// Helper function to create test session with mock connection
func createMockPgSession(t *testing.T, sessionID string, mockConn *MockPgConn) *PgSession {
	session, err := NewPgSession(context.Background(), PgSessionParams{
		SessionID:     sessionID,
		SessionTable:  "test_sessions",
		MessagesTable: "test_messages",
		Conn:          mockConn,
	})
	require.NoError(t, err)
	return session
}

// Helper function to create test message items
func createTestItems() []TResponseInputItem {
	return []TResponseInputItem{
		{OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Hello")},
			Role:    responses.EasyInputMessageRoleUser,
			Type:    responses.EasyInputMessageTypeMessage,
		}},
		{OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Hi there!")},
			Role:    responses.EasyInputMessageRoleAssistant,
			Type:    responses.EasyInputMessageTypeMessage,
		}},
		{OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("How are you?")},
			Role:    responses.EasyInputMessageRoleUser,
			Type:    responses.EasyInputMessageTypeMessage,
		}},
	}
}

func TestPgSession_NewPgSession(t *testing.T) {
	ctx := context.Background()

	t.Run("missing connection string and no conn provided", func(t *testing.T) {
		_, err := NewPgSession(ctx, PgSessionParams{
			SessionID: "test",
		})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "connection string is required")
	})

	t.Run("successful creation with mock connection", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock the initDB calls
		mockConn.On("Exec", mock.Anything, mock.MatchedBy(func(sql string) bool {
			return fmt.Sprintf(sql, "test_sessions") != ""
		})).Return(nil, nil).Once()

		mockConn.On("Exec", mock.Anything, mock.MatchedBy(func(sql string) bool {
			return fmt.Sprintf(sql, "test_messages", "test_sessions") != ""
		})).Return(nil, nil).Once()

		mockConn.On("Exec", mock.Anything, mock.MatchedBy(func(sql string) bool {
			return fmt.Sprintf(sql, "test_messages", "test_messages") != ""
		})).Return(nil, nil).Once()

		session, err := NewPgSession(ctx, PgSessionParams{
			SessionID:     "test",
			SessionTable:  "test_sessions",
			MessagesTable: "test_messages",
			Conn:          mockConn,
		})
		require.NoError(t, err)

		assert.Equal(t, "test", session.SessionID(ctx))
		assert.Equal(t, "test_sessions", session.sessionTable)
		assert.Equal(t, "test_messages", session.messagesTable)

		mockConn.AssertExpectations(t)
	})
}

func TestPgSession_GetItems(t *testing.T) {
	ctx := context.Background()

	t.Run("no limit - empty session", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock empty query result
		mockRows := NewMockPgRows([]string{})
		mockConn.On("Query", mock.Anything, mock.AnythingOfType("string"), "test").Return(mockRows, nil)

		session := createMockPgSession(t, "test", mockConn)

		items, err := session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Empty(t, items)

		mockConn.AssertExpectations(t)
	})

	t.Run("no limit - with items", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Create test data
		testItems := createTestItems()
		var jsonData []string
		for _, item := range testItems {
			jsonBytes, _ := item.MarshalJSON()
			jsonData = append(jsonData, string(jsonBytes))
		}

		mockRows := NewMockPgRows(jsonData)
		mockConn.On("Query", mock.Anything, mock.AnythingOfType("string"), "test").Return(mockRows, nil)

		session := createMockPgSession(t, "test", mockConn)

		items, err := session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Len(t, items, 3)
		assert.Equal(t, testItems, items)

		mockConn.AssertExpectations(t)
	})

	t.Run("with limit", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Create test data - should return last 2 items in reverse order due to DESC then reverse
		testItems := createTestItems()
		var jsonData []string
		// Mock returns items in DESC order (last 2 items)
		for i := len(testItems) - 1; i >= len(testItems)-2; i-- {
			jsonBytes, _ := testItems[i].MarshalJSON()
			jsonData = append(jsonData, string(jsonBytes))
		}

		mockRows := NewMockPgRows(jsonData)
		mockConn.On("Query", mock.Anything, mock.AnythingOfType("string"), "test", 2).Return(mockRows, nil)

		session := createMockPgSession(t, "test", mockConn)

		items, err := session.GetItems(ctx, 2)
		require.NoError(t, err)
		assert.Len(t, items, 2)
		// Should be last 2 items in chronological order
		assert.Equal(t, testItems[1:], items)

		mockConn.AssertExpectations(t)
	})
}

func TestPgSession_AddItems(t *testing.T) {
	ctx := context.Background()

	t.Run("empty items list", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		session := createMockPgSession(t, "test", mockConn)

		err := session.AddItems(ctx, []TResponseInputItem{})
		assert.NoError(t, err)

		mockConn.AssertExpectations(t)
	})

	t.Run("single item", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock session creation
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Once()

		// Mock item insertion
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test", mock.AnythingOfType("string")).Return(nil, nil).Once()

		// Mock timestamp update
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Once()

		session := createMockPgSession(t, "test", mockConn)

		testItems := createTestItems()
		err := session.AddItems(ctx, testItems[:1])
		require.NoError(t, err)

		mockConn.AssertExpectations(t)
	})

	t.Run("multiple items", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock session creation
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Once()

		// Mock item insertions (3 items)
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test", mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock timestamp update
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Once()

		session := createMockPgSession(t, "test", mockConn)

		testItems := createTestItems()
		err := session.AddItems(ctx, testItems)
		require.NoError(t, err)

		mockConn.AssertExpectations(t)
	})
}

func TestPgSession_PopItem(t *testing.T) {
	ctx := context.Background()

	t.Run("from empty session", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock empty result (no rows)
		mockRow := NewMockPgRow("", true)
		mockConn.On("QueryRow", mock.Anything, mock.AnythingOfType("string"), "test").Return(mockRow)

		session := createMockPgSession(t, "test", mockConn)

		item, err := session.PopItem(ctx)
		require.NoError(t, err)
		assert.Nil(t, item)

		mockConn.AssertExpectations(t)
	})

	t.Run("from session with items", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Create test data
		testItems := createTestItems()
		lastItem := testItems[len(testItems)-1]
		jsonBytes, _ := lastItem.MarshalJSON()

		mockRow := NewMockPgRow(string(jsonBytes), false)
		mockConn.On("QueryRow", mock.Anything, mock.AnythingOfType("string"), "test").Return(mockRow)

		session := createMockPgSession(t, "test", mockConn)

		item, err := session.PopItem(ctx)
		require.NoError(t, err)
		require.NotNil(t, item)
		assert.Equal(t, lastItem, *item)

		mockConn.AssertExpectations(t)
	})
}

func TestPgSession_ClearSession(t *testing.T) {
	ctx := context.Background()

	t.Run("clear session", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock clear operations
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Times(2)

		session := createMockPgSession(t, "test", mockConn)

		err := session.ClearSession(ctx)
		require.NoError(t, err)

		mockConn.AssertExpectations(t)
	})
}

func TestPgSession_OutputMessage(t *testing.T) {
	ctx := context.Background()

	t.Run("output message handling", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock session creation
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Once()

		// Mock item insertion
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test", mock.AnythingOfType("string")).Return(nil, nil).Once()

		// Mock timestamp update
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Once()

		session := createMockPgSession(t, "test", mockConn)

		// Test with output message (similar to SQLite test for unmarshalMessageData fix)
		outputItem := TResponseInputItem{
			OfOutputMessage: &responses.ResponseOutputMessageParam{
				ID: "msg_123",
				Content: []responses.ResponseOutputMessageContentUnionParam{
					{OfOutputText: &responses.ResponseOutputTextParam{
						Text: "Output message test",
						Type: constant.ValueOf[constant.OutputText](),
					}},
				},
				Status: responses.ResponseOutputMessageStatusCompleted,
				Role:   constant.ValueOf[constant.Assistant](),
				Type:   constant.ValueOf[constant.Message](),
			},
		}

		err := session.AddItems(ctx, []TResponseInputItem{outputItem})
		require.NoError(t, err)

		mockConn.AssertExpectations(t)
	})
}

func TestPgSession_Concurrency(t *testing.T) {
	ctx := context.Background()

	t.Run("concurrent operations", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// For concurrent operations, we expect multiple session creation and item insertion calls
		const numGoroutines = 5
		const itemsPerGoroutine = 10

		// Mock session creation calls (may be called multiple times due to concurrency)
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Maybe()

		// Mock item insertions (50 total items)
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test", mock.AnythingOfType("string")).Return(nil, nil).Times(numGoroutines * itemsPerGoroutine)

		// Mock timestamp updates (may be called multiple times)
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, nil).Maybe()

		session := createMockPgSession(t, "test", mockConn)

		// Test concurrent AddItems calls
		done := make(chan error, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(id int) {
				var items []TResponseInputItem
				for j := 0; j < itemsPerGoroutine; j++ {
					items = append(items, TResponseInputItem{
						OfMessage: &responses.EasyInputMessageParam{
							Content: responses.EasyInputMessageContentUnionParam{
								OfString: param.NewOpt(fmt.Sprintf("Goroutine %d, Item %d", id, j)),
							},
							Role: responses.EasyInputMessageRoleUser,
							Type: responses.EasyInputMessageTypeMessage,
						},
					})
				}
				done <- session.AddItems(ctx, items)
			}(i)
		}

		// Wait for all goroutines to complete
		for i := 0; i < numGoroutines; i++ {
			assert.NoError(t, <-done)
		}

		// Note: We can't assert exact call counts due to concurrency and the way mocks work
		// The important thing is that no errors occurred and all operations completed
	})
}

func TestPgSession_ErrorHandling(t *testing.T) {
	ctx := context.Background()

	t.Run("query error", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock query error
		mockConn.On("Query", mock.Anything, mock.AnythingOfType("string"), "test").Return((*MockPgRows)(nil), fmt.Errorf("database error"))

		session := createMockPgSession(t, "test", mockConn)

		_, err := session.GetItems(ctx, 0)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "error querying session items")

		mockConn.AssertExpectations(t)
	})

	t.Run("exec error in AddItems", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock session creation error
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string"), "test").Return(nil, fmt.Errorf("database error"))

		session := createMockPgSession(t, "test", mockConn)

		testItems := createTestItems()
		err := session.AddItems(ctx, testItems[:1])
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "error ensuring session exists")

		mockConn.AssertExpectations(t)
	})

	t.Run("invalid JSON in unmarshalMessageData", func(t *testing.T) {
		mockConn := &MockPgConn{}

		// Mock initDB calls
		mockConn.On("Exec", mock.Anything, mock.AnythingOfType("string")).Return(nil, nil).Times(3)

		// Mock query with invalid JSON
		mockRows := NewMockPgRows([]string{"invalid json"})
		mockConn.On("Query", mock.Anything, mock.AnythingOfType("string"), "test").Return(mockRows, nil)

		session := createMockPgSession(t, "test", mockConn)

		items, err := session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Empty(t, items) // Invalid JSON should be skipped

		mockConn.AssertExpectations(t)
	})
}
