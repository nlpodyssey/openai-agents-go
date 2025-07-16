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
	"testing"

	"github.com/modelcontextprotocol/go-sdk/jsonschema"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunnerCallsMCPTool(t *testing.T) {
	// Test that the runner calls an MCP tool when the model produces a tool call.
	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			server := agentstesting.NewFakeMCPServer(nil, nil, "")
			server.AddTool("test_tool_1", nil)
			server.AddTool("test_tool_2", nil)
			server.AddTool("test_tool_3", nil)

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model).AddMCPServer(server)

			model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
				// First turn: a message and tool call
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("a_message"),
					agentstesting.GetFunctionToolCall("test_tool_2", ""),
				}},
				// Second turn: text message
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("done"),
				}},
			})

			if streaming {
				result, err := agents.RunStreamed(t.Context(), agent, "user_message")
				require.NoError(t, err)
				err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
				require.NoError(t, err)
			} else {
				_, err := agents.Run(t.Context(), agent, "user_message")
				require.NoError(t, err)
			}
			assert.Equal(t, []string{"test_tool_2"}, server.ToolCalls)
		})
	}
}

func TestRunnerAssertsWhenMCPToolNotFound(t *testing.T) {
	// Test that the runner asserts when an MCP tool is not found.
	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			server := agentstesting.NewFakeMCPServer(nil, nil, "")
			server.AddTool("test_tool_1", nil)
			server.AddTool("test_tool_2", nil)
			server.AddTool("test_tool_3", nil)

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model).AddMCPServer(server)

			model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
				// First turn: a message and tool call
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("a_message"),
					agentstesting.GetFunctionToolCall("test_tool_doesnt_exist", ""),
				}},
				// Second turn: text message
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("done"),
				}},
			})

			var finalError error
			if streaming {
				result, err := agents.RunStreamed(t.Context(), agent, "user_message")
				require.NoError(t, err)
				finalError = result.StreamEvents(func(agents.StreamEvent) error { return nil })
			} else {
				_, finalError = agents.Run(t.Context(), agent, "user_message")
			}
			require.ErrorAs(t, finalError, &agents.ModelBehaviorError{})
		})
	}
}

func TestRunnerWorksWithMultipleMCPServers(t *testing.T) {
	// Test that the runner works with multiple MCP servers.
	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			server1 := agentstesting.NewFakeMCPServer(nil, nil, "server_1")
			server1.AddTool("test_tool_1", nil)

			server2 := agentstesting.NewFakeMCPServer(nil, nil, "server_2")
			server2.AddTool("test_tool_2", nil)
			server2.AddTool("test_tool_3", nil)

			servers := []agents.MCPServer{server1, server2}

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model).WithMCPServers(servers)

			model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
				// First turn: a message and tool call
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("a_message"),
					agentstesting.GetFunctionToolCall("test_tool_2", ""),
				}},
				// Second turn: text message
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("done"),
				}},
			})

			if streaming {
				result, err := agents.RunStreamed(t.Context(), agent, "user_message")
				require.NoError(t, err)
				err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
				require.NoError(t, err)
			} else {
				_, err := agents.Run(t.Context(), agent, "user_message")
				require.NoError(t, err)
			}
			assert.Empty(t, server1.ToolCalls)
			assert.Equal(t, []string{"test_tool_2"}, server2.ToolCalls)
		})
	}
}

func TestRunnerErrorsWhenMCPToolsClash(t *testing.T) {
	// Test that the runner errors when multiple servers have the same tool name.
	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			server1 := agentstesting.NewFakeMCPServer(nil, nil, "server_1")
			server1.AddTool("test_tool_1", nil)
			server1.AddTool("test_tool_2", nil)

			server2 := agentstesting.NewFakeMCPServer(nil, nil, "server_2")
			server2.AddTool("test_tool_2", nil)
			server2.AddTool("test_tool_3", nil)

			servers := []agents.MCPServer{server1, server2}

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model).WithMCPServers(servers)

			model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
				// First turn: a message and tool call
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("a_message"),
					agentstesting.GetFunctionToolCall("test_tool_3", ""),
				}},
				// Second turn: text message
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("done"),
				}},
			})

			var finalError error
			if streaming {
				result, err := agents.RunStreamed(t.Context(), agent, "user_message")
				require.NoError(t, err)
				finalError = result.StreamEvents(func(agents.StreamEvent) error { return nil })
			} else {
				_, finalError = agents.Run(t.Context(), agent, "user_message")
			}
			require.ErrorAs(t, finalError, &agents.UserError{})
		})
	}
}

func TestRunnerCallsMCPToolWithArgs(t *testing.T) {
	// Test that the runner calls an MCP tool when the model produces a tool call.
	for _, streaming := range []bool{true, false} {
		t.Run(fmt.Sprintf("streaming %v", streaming), func(t *testing.T) {
			server := agentstesting.NewFakeMCPServer(nil, nil, "")

			ctx := t.Context()
			require.NoError(t, server.Connect(ctx))
			t.Cleanup(func() {
				assert.NoError(t, server.Cleanup(ctx))
			})

			server.AddTool("test_tool_1", nil)
			server.AddTool("test_tool_2", &jsonschema.Schema{
				Type:     "object",
				Required: []string{"bar", "baz"},
				Properties: map[string]*jsonschema.Schema{
					"bar": {Type: "string"},
					"baz": {Type: "integer"},
				},
			})
			server.AddTool("test_tool_3", nil)

			model := agentstesting.NewFakeModel(false, nil)
			agent := agents.New("test").WithModelInstance(model).AddMCPServer(server)

			jsonArgs := `{"bar":"baz","baz":1}`

			model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
				// First turn: a message and tool call
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("a_message"),
					agentstesting.GetFunctionToolCall("test_tool_2", jsonArgs),
				}},
				// Second turn: text message
				{Value: []agents.TResponseOutputItem{
					agentstesting.GetTextMessage("done"),
				}},
			})

			if streaming {
				result, err := agents.RunStreamed(t.Context(), agent, "user_message")
				require.NoError(t, err)
				err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
				require.NoError(t, err)
			} else {
				_, err := agents.Run(t.Context(), agent, "user_message")
				require.NoError(t, err)
			}

			assert.Equal(t, []string{"test_tool_2"}, server.ToolCalls)
			assert.Equal(t, []string{`result_test_tool_2_{"bar":"baz","baz":1}`}, server.ToolResults)
		})
	}
}
