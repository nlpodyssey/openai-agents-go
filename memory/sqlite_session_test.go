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
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSQLiteSession_GetItems(t *testing.T) {
	ctx := t.Context()

	t.Run("no limit", func(t *testing.T) {
		session, err := NewSQLiteSession(ctx, SQLiteSessionParams{
			SessionID:        "test",
			DBDataSourceName: filepath.Join(t.TempDir(), "test.db"),
		})
		require.NoError(t, err)
		t.Cleanup(func() { assert.NoError(t, session.Close()) })

		// Test adding and retrieving items
		items := []TResponseInputItem{
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

		// Add first two items
		require.NoError(t, session.AddItems(ctx, items[:2]))
		retrieved, err := session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Equal(t, items[:2], retrieved)

		// Add another item
		require.NoError(t, session.AddItems(ctx, items[2:]))
		retrieved, err = session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Equal(t, items, retrieved)

		// Test clearing session
		require.NoError(t, session.ClearSession(ctx))
		retrieved, err = session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Empty(t, retrieved)
	})

	t.Run("with limit", func(t *testing.T) {
		session, err := NewSQLiteSession(ctx, SQLiteSessionParams{
			SessionID:        "test",
			DBDataSourceName: filepath.Join(t.TempDir(), "test.db"),
		})
		require.NoError(t, err)
		t.Cleanup(func() { assert.NoError(t, session.Close()) })

		// Add multiple items
		items := []TResponseInputItem{
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Message 1")},
				Role:    responses.EasyInputMessageRoleUser,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Response 1")},
				Role:    responses.EasyInputMessageRoleAssistant,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Message 2")},
				Role:    responses.EasyInputMessageRoleUser,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Response 2")},
				Role:    responses.EasyInputMessageRoleAssistant,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Message 3")},
				Role:    responses.EasyInputMessageRoleUser,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Response 3")},
				Role:    responses.EasyInputMessageRoleAssistant,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
		}
		require.NoError(t, session.AddItems(ctx, items))

		// Test getting all items (default behavior)
		allItems, err := session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Equal(t, items, allItems)

		// Test getting latest 2 items
		latest2, err := session.GetItems(ctx, 2)
		require.NoError(t, err)
		assert.Equal(t, items[4:], latest2)

		// Test getting latest 4 items
		latest4, err := session.GetItems(ctx, 4)
		require.NoError(t, err)
		assert.Equal(t, items[2:], latest4)

		// Test getting more items than available
		latest10, err := session.GetItems(ctx, 10)
		require.NoError(t, err)
		assert.Equal(t, items, latest10) // Should return all available items

		// Test negative limit same as zero (all items)
		allItems, err = session.GetItems(ctx, -123)
		require.NoError(t, err)
		assert.Equal(t, items, allItems)
	})
}

func TestSQLiteSession_PopItem(t *testing.T) {
	ctx := t.Context()

	t.Run("from same session", func(t *testing.T) {
		session, err := NewSQLiteSession(ctx, SQLiteSessionParams{
			SessionID:        "pop_test",
			DBDataSourceName: filepath.Join(t.TempDir(), "pop_test.db"),
		})
		require.NoError(t, err)
		t.Cleanup(func() { assert.NoError(t, session.Close()) })

		// Test popping from empty session
		popped, err := session.PopItem(ctx)
		require.NoError(t, err)
		assert.Nil(t, popped)

		// Add items
		items := []TResponseInputItem{
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
		require.NoError(t, session.AddItems(ctx, items))

		// Verify all items are there
		retrieved, err := session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Equal(t, items, retrieved)

		// Pop the most recent item
		popped, err = session.PopItem(ctx)
		require.NoError(t, err)
		assert.Equal(t, &items[2], popped)

		// Verify item was removed
		retrieved, err = session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Equal(t, items[:2], retrieved)

		// Pop another item
		popped, err = session.PopItem(ctx)
		require.NoError(t, err)
		assert.Equal(t, &items[1], popped)

		// Verify item was removed
		retrieved, err = session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Equal(t, items[:1], retrieved)

		// Pop the last item
		popped, err = session.PopItem(ctx)
		require.NoError(t, err)
		assert.Equal(t, &items[0], popped)

		// Verify session is empty
		retrieved, err = session.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Empty(t, retrieved)

		// Try to pop from empty session again
		popped, err = session.PopItem(ctx)
		require.NoError(t, err)
		assert.Nil(t, popped)
	})

	t.Run("different sessions", func(t *testing.T) {
		// Test that PopItem only affects the specified session.
		dbPath := filepath.Join(t.TempDir(), "pop_sessions_test.db")

		session1, err := NewSQLiteSession(ctx, SQLiteSessionParams{
			SessionID:        "session_1",
			DBDataSourceName: dbPath,
		})
		require.NoError(t, err)
		t.Cleanup(func() { assert.NoError(t, session1.Close()) })

		session2, err := NewSQLiteSession(ctx, SQLiteSessionParams{
			SessionID:        "session_2",
			DBDataSourceName: dbPath,
		})
		require.NoError(t, err)
		t.Cleanup(func() { assert.NoError(t, session2.Close()) })

		// Add items to both sessions
		items1 := []TResponseInputItem{
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Session 1 message")},
				Role:    responses.EasyInputMessageRoleUser,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
		}
		items2 := []TResponseInputItem{
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Session 2 message 1")},
				Role:    responses.EasyInputMessageRoleUser,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
			{OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Session 2 message 2")},
				Role:    responses.EasyInputMessageRoleUser,
				Type:    responses.EasyInputMessageTypeMessage,
			}},
		}

		require.NoError(t, session1.AddItems(ctx, items1))
		require.NoError(t, session2.AddItems(ctx, items2))

		// Pop from session 2
		popped, err := session2.PopItem(ctx)
		require.NoError(t, err)
		assert.Equal(t, &items2[1], popped)

		// Verify session 1 is unaffected
		session1Items, err := session1.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Equal(t, items1, session1Items)

		// Verify session 2 has one item left
		session2Items, err := session2.GetItems(ctx, 0)
		require.NoError(t, err)
		assert.Equal(t, items2[:1], session2Items)
	})
}

func Test_unmarshalMessageData(t *testing.T) {
	t.Run("incorrect output messages unmarshaling fix", func(t *testing.T) {
		toMarshal := responses.ResponseInputItemUnionParam{
			OfOutputMessage: &responses.ResponseOutputMessageParam{
				ID: "msg_123",
				Content: []responses.ResponseOutputMessageContentUnionParam{
					{OfOutputText: &responses.ResponseOutputTextParam{
						Text: "Foo",
						Type: constant.ValueOf[constant.OutputText](),
					}},
				},
				Status: responses.ResponseOutputMessageStatusCompleted,
				Role:   constant.ValueOf[constant.Assistant](),
				Type:   constant.ValueOf[constant.Message](),
			},
		}

		marshaled, err := json.Marshal(toMarshal)
		require.NoError(t, err)

		t.Run("the issue exists with normal JSON unmarshaling", func(t *testing.T) {
			// Make sure that the problem targeted by the fix exists: upon dependency updates,
			// the situation might change and the fix might become obsolete.
			var result responses.ResponseInputItemUnionParam
			require.NoError(t, json.Unmarshal(marshaled, &result))
			assert.Equal(t, responses.ResponseInputItemUnionParam{
				OfMessage: &responses.EasyInputMessageParam{
					// Here's the problem: the original output text content is lost!
					Content: responses.EasyInputMessageContentUnionParam{},
					Role:    responses.EasyInputMessageRoleAssistant,
					Type:    responses.EasyInputMessageTypeMessage,
				},
			}, result)
		})

		t.Run("the fix works", func(t *testing.T) {
			result, err := unmarshalMessageData(string(marshaled))
			require.NoError(t, err)
			assert.Equal(t, toMarshal, result)
		})
	})
}
