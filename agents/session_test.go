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

package agents_test

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/memory"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgentSession(t *testing.T) {
	runAgent := func(t *testing.T, streaming bool, session memory.Session, agent *agents.Agent, input string) any {
		t.Helper()

		runner := agents.Runner{
			Config: agents.RunConfig{
				Session: session,
			},
		}

		var finalOutput any
		if streaming {
			result, err := runner.RunStreamed(t.Context(), agent, input)
			require.NoError(t, err)
			err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
			require.NoError(t, err)
			finalOutput = result.FinalOutput()
		} else {
			result, err := runner.Run(t.Context(), agent, input)
			require.NoError(t, err)
			finalOutput = result.FinalOutput
		}
		return finalOutput
	}

	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			t.Run("basic functionality", func(t *testing.T) {
				// Test basic session memory functionality with SQLite backend.
				session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
					SessionID:        "test",
					DBDataSourceName: filepath.Join(t.TempDir(), "test.db"),
				})
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, session.Close()) })

				model := agentstesting.NewFakeModel(false, nil)
				agent := agents.New("test").WithModelInstance(model)

				// First turn
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("San Francisco")},
				})
				result1 := runAgent(t, streaming, session, agent, "What city is the Golden Gate Bridge in?")
				assert.Equal(t, "San Francisco", result1)

				// Second turn - should have conversation history
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("California")},
				})
				result2 := runAgent(t, streaming, session, agent, "What state is it in?")
				assert.Equal(t, "California", result2)

				// Verify that the input to the second turn includes the previous conversation
				// The model should have received the full conversation history
				lastInput := model.LastTurnArgs.Input
				require.IsType(t, agents.InputItems{}, lastInput)
				assert.Len(t, lastInput.(agents.InputItems), 3)
			})

			t.Run("no session", func(t *testing.T) {
				// Test that session memory is disabled when Session is nil.
				model := agentstesting.NewFakeModel(false, nil)
				agent := agents.New("test").WithModelInstance(model)

				// First turn (no session parameters = disabled)
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Hello")},
				})
				result1 := runAgent(t, streaming, nil, agent, "Hi there")
				assert.Equal(t, "Hello", result1)

				// Second turn - should NOT have conversation history
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("I don't remember")},
				})
				result2 := runAgent(t, streaming, nil, agent, "Do you remember what I said?")
				assert.Equal(t, "I don't remember", result2)

				// Verify that the input to the second turn is just the current message
				lastInput := model.LastTurnArgs.Input
				require.IsType(t, agents.InputItems{}, lastInput)
				assert.Len(t, lastInput.(agents.InputItems), 1)
			})

			t.Run("different sessions", func(t *testing.T) {
				// Test that different session IDs maintain separate conversation histories.
				dbPath := filepath.Join(t.TempDir(), "test.db")

				model := agentstesting.NewFakeModel(false, nil)
				agent := agents.New("test").WithModelInstance(model)

				// Session 1
				session1, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
					SessionID:        "session_1",
					DBDataSourceName: dbPath,
				})
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, session1.Close()) })

				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("I like cats")},
				})
				result1 := runAgent(t, streaming, session1, agent, "I like cats")
				assert.Equal(t, "I like cats", result1)

				// Session 2 - different session
				session2, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
					SessionID:        "session_2",
					DBDataSourceName: dbPath,
				})
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, session2.Close()) })

				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("I like dogs")},
				})
				result2 := runAgent(t, streaming, session2, agent, "I like dogs")
				assert.Equal(t, "I like dogs", result2)

				// Back to Session 1 - should remember cats, not dogs
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Yes, you mentioned cats")},
				})
				result3 := runAgent(t, streaming, session1, agent, "What did I say I like?")
				assert.Equal(t, "Yes, you mentioned cats", result3)

				lastInput := model.LastTurnArgs.Input
				require.IsType(t, agents.InputItems{}, lastInput)
				require.Len(t, lastInput.(agents.InputItems), 3)
				assert.Equal(t, "I like cats", lastInput.(agents.InputItems)[0].OfMessage.Content.OfString.Value)

				// Back to Session 2 - should remember dogs, not cats
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("Yes, you mentioned dogs")},
				})
				result4 := runAgent(t, streaming, session2, agent, "What did I say I like?")
				assert.Equal(t, "Yes, you mentioned dogs", result4)

				lastInput = model.LastTurnArgs.Input
				require.IsType(t, agents.InputItems{}, lastInput)
				require.Len(t, lastInput.(agents.InputItems), 3)
				assert.Equal(t, "I like dogs", lastInput.(agents.InputItems)[0].OfMessage.Content.OfString.Value)
			})

			t.Run("cannot use both session and list input items", func(t *testing.T) {
				// Test that passing both a session and list input raises a UserError.
				session, err := memory.NewSQLiteSession(t.Context(), memory.SQLiteSessionParams{
					SessionID:        "test",
					DBDataSourceName: filepath.Join(t.TempDir(), "test.db"),
				})
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, session.Close()) })

				model := agentstesting.NewFakeModel(false, nil)
				agent := agents.New("test").WithModelInstance(model)

				// Test that providing both a session and a list input raises a UserError
				model.SetNextOutput(agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("This shouldn't run")},
				})

				listInput := agents.InputItems{
					{OfMessage: &responses.EasyInputMessageParam{
						Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt("Test message")},
						Role:    responses.EasyInputMessageRoleUser,
						Type:    responses.EasyInputMessageTypeMessage,
					}},
				}

				runner := agents.Runner{
					Config: agents.RunConfig{
						Session: session,
					},
				}

				var finalError error
				if streaming {
					result, err := runner.RunInputsStreamed(t.Context(), agent, listInput)
					require.NoError(t, err)
					finalError = result.StreamEvents(func(agents.StreamEvent) error { return nil })
				} else {
					_, finalError = runner.RunInputs(t.Context(), agent, listInput)
				}

				assert.ErrorAs(t, finalError, &agents.UserError{})

				// Verify the error message explains the issue
				assert.ErrorContains(t, finalError, "Cannot provide both a session and a list of input items")
				assert.ErrorContains(t, finalError, "manually manage conversation history")
			})
		})
	}
}
