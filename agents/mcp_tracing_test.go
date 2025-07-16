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
	"cmp"
	"slices"
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/tracing/tracingtesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPTracing(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(false, nil)
	server := agentstesting.NewFakeMCPServer(nil, nil, "")
	server.AddTool("test_tool_1", nil)
	agent := agents.New("test").
		WithModelInstance(model).
		AddMCPServer(server).
		WithTools(agentstesting.GetFunctionTool("non_mcp_tool", "tool_result"))

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("test_tool_1", ""),
		}},
		// Second turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	ctx := t.Context()
	type m = map[string]any

	// First run: should list MCP tools before first and second steps
	streamResult, err := agents.RunStreamed(ctx, agent, "first_test")
	require.NoError(t, err)
	err = streamResult.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "done", streamResult.FinalOutput())

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	// Should have a single tool listing, and the function span should have MCP data
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "mcp_tools",
					"data": m{
						"server": "fake_mcp_server",
						"result": []string{"test_tool_1"},
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test",
						"handoffs":    []string{},
						"tools":       []string{"test_tool_1", "non_mcp_tool"},
						"output_type": "string",
					},
					"children": []m{
						{
							"type": "function",
							"data": m{
								"name":     "test_tool_1",
								"output":   `{"type":"text","text":"result_test_tool_1_null"}`,
								"mcp_data": m{"server": "fake_mcp_server"},
							},
						},
						{
							"type": "mcp_tools",
							"data": m{"server": "fake_mcp_server", "result": []string{"test_tool_1"}},
						},
					},
				},
			},
		},
	}, spans)

	server.AddTool("test_tool_2", nil)

	tracingtesting.SpanProcessorTesting().Clear()

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool calls
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("non_mcp_tool", ""),
			agentstesting.GetFunctionToolCall("test_tool_2", ""),
		}},
		// Second turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	_, err = agents.Run(ctx, agent, "second_test")
	require.NoError(t, err)

	spans = tracingtesting.FetchNormalizedSpans(t, false, false, false)

	// Tool calls from first turn run concurrently: let's sort the spans in a deterministic way
	children := spans[0]["children"].([]m)[1]["children"].([]m)
	slices.SortFunc(children, func(a, b m) int {
		aData := a["data"].(m)
		bData := b["data"].(m)

		aName, _ := aData["name"].(string)
		bName, _ := bData["name"].(string)

		aServer, _ := aData["server"].(string)
		bServer, _ := bData["server"].(string)

		return strings.Compare(
			cmp.Or(aName, aServer),
			cmp.Or(bName, bServer),
		)
	})

	// Should have a single tool listing, and the function span should have MCP data, and the non-mcp
	// tool function span should not have MCP data
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "mcp_tools",
					"data": m{
						"server": "fake_mcp_server",
						"result": []string{"test_tool_1", "test_tool_2"},
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test",
						"handoffs":    []string{},
						"tools":       []string{"test_tool_1", "test_tool_2", "non_mcp_tool"},
						"output_type": "string",
					},
					"children": []m{
						{
							"type": "mcp_tools",
							"data": m{
								"server": "fake_mcp_server",
								"result": []string{"test_tool_1", "test_tool_2"},
							},
						},
						{
							"type": "function",
							"data": m{
								"name":   "non_mcp_tool",
								"output": "tool_result",
							},
						},
						{
							"type": "function",
							"data": m{
								"name":     "test_tool_2",
								"output":   `{"type":"text","text":"result_test_tool_2_null"}`,
								"mcp_data": m{"server": "fake_mcp_server"},
							},
						},
					},
				},
			},
		},
	}, spans)

	tracingtesting.SpanProcessorTesting().Clear()

	// Add more tools to the server
	server.AddTool("test_tool_3", nil)

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

	_, err = agents.Run(ctx, agent, "third_test")
	require.NoError(t, err)

	spans = tracingtesting.FetchNormalizedSpans(t, false, false, false)

	// Should have a single tool listing, and the function span should have MCP data, and the non-mcp
	// tool function span should not have MCP data
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "mcp_tools",
					"data": m{
						"server": "fake_mcp_server",
						"result": []string{"test_tool_1", "test_tool_2", "test_tool_3"},
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test",
						"handoffs":    []string{},
						"tools":       []string{"test_tool_1", "test_tool_2", "test_tool_3", "non_mcp_tool"},
						"output_type": "string",
					},
					"children": []m{
						{
							"type": "function",
							"data": m{
								"name":     "test_tool_3",
								"output":   `{"type":"text","text":"result_test_tool_3_null"}`,
								"mcp_data": m{"server": "fake_mcp_server"},
							},
						},
						{
							"type": "mcp_tools",
							"data": m{
								"server": "fake_mcp_server",
								"result": []string{"test_tool_1", "test_tool_2", "test_tool_3"},
							},
						},
					},
				},
			},
		},
	}, spans)
}
