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

package agents

import (
	"testing"

	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmptyResponseIsFinalOutput(t *testing.T) {
	agent := &Agent{Name: "test"}
	response := ModelResponse{
		Output:     nil,
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})

	assert.Equal(t, InputString("hello"), result.OriginalInput)
	assert.Empty(t, result.GeneratedItems())

	require.IsType(t, NextStepFinalOutput{}, result.NextStep)
	assert.Equal(t, "", result.NextStep.(NextStepFinalOutput).Output)
}

func TestPlaintextAgentNoToolCallsIsFinalOutput(t *testing.T) {
	agent := &Agent{Name: "test"}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("hello_world"),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})

	assert.Equal(t, InputString("hello"), result.OriginalInput)
	require.Len(t, result.GeneratedItems(), 1)
	assertItemIsMessage(t, result.GeneratedItems()[0], agent, "hello_world")

	require.IsType(t, NextStepFinalOutput{}, result.NextStep)
	assert.Equal(t, "hello_world", result.NextStep.(NextStepFinalOutput).Output)
}

func TestPlaintextAgentNoToolCallsMultipleMessagesIsFinalOutput(t *testing.T) {
	agent := &Agent{Name: "test"}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("hello_world"),
			getTextMessage("bye"),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
		originalInput: InputItems{
			getTextInputItem("test"),
			getTextInputItem("test2"),
		},
	})

	require.Len(t, result.OriginalInput, 2)
	require.Len(t, result.GeneratedItems(), 2)
	assertItemIsMessage(t, result.GeneratedItems()[0], agent, "hello_world")
	assertItemIsMessage(t, result.GeneratedItems()[1], agent, "bye")

	require.IsType(t, NextStepFinalOutput{}, result.NextStep)
	assert.Equal(t, "bye", result.NextStep.(NextStepFinalOutput).Output)
}

func TestPlaintextAgentWithToolCallIsRunAgain(t *testing.T) {
	agent := &Agent{
		Name: "test",
		Tools: []Tool{
			getFunctionTool("test", "123"),
		},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("hello_world"),
			getFunctionToolCall("test", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})

	assert.Equal(t, InputString("hello"), result.OriginalInput)

	// 3 items: new message, tool call, tool result
	items := result.GeneratedItems()
	require.Len(t, items, 3)

	assertItemIsMessage(t, items[0], agent, "hello_world")
	assertItemIsFunctionToolCall(t, items[1], agent, "test", "")
	assertItemIsFunctionToolCallOutput(t, items[2], agent, "123")

	assert.IsType(t, NextStepRunAgain{}, result.NextStep)
}

func TestRunStepExecutionMultipleToolCalls(t *testing.T) {
	agent := &Agent{
		Name: "test",
		Tools: []Tool{
			getFunctionTool("test_1", "123"),
			getFunctionTool("test_2", "456"),
			getFunctionTool("test_3", "789"),
		},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getFunctionToolCall("test_1", ""),
			getFunctionToolCall("test_2", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})

	assert.Equal(t, InputString("hello"), result.OriginalInput)

	// 5 items: new message, 2 tool calls, 2 tool call outputs
	items := result.GeneratedItems()
	require.Len(t, items, 5)

	assertItemIsMessage(t, items[0], agent, "Hello, world!")
	assertItemIsFunctionToolCall(t, items[1], agent, "test_1", "")
	assertItemIsFunctionToolCall(t, items[2], agent, "test_2", "")
	assertItemIsFunctionToolCallOutput(t, items[3], agent, "123")
	assertItemIsFunctionToolCallOutput(t, items[4], agent, "456")

	assert.IsType(t, NextStepRunAgain{}, result.NextStep)
}

func TestHandoffOutputLeadsToHandoffNextStep(t *testing.T) {
	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:          "test_3",
		AgentHandoffs: []*Agent{agent1, agent2},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getHandoffToolCall(agent1, "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent3,
		response: response,
	})

	require.IsType(t, NextStepHandoff{}, result.NextStep)
	assert.Same(t, agent1, result.NextStep.(NextStepHandoff).NewAgent)

	assert.Len(t, result.GeneratedItems(), 3)
}

func TestFinalOutputWithoutToolRunsAgain(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	agent := &Agent{
		Name:       "test",
		OutputType: OutputType[Foo](),
		Tools: []Tool{
			getFunctionTool("tool_1", "result"),
		},
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFunctionToolCall("tool_1", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})
	assert.IsType(t, NextStepRunAgain{}, result.NextStep)
	assert.Len(t, result.GeneratedItems(), 2, "expected 2 items: tool call, tool call output")
}

func TestFinalOutputLeadsToFinalOutputNextStep(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	agent := &Agent{
		Name:       "test",
		OutputType: OutputType[Foo](),
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getTextMessage("Hello, world!"),
			getFinalOutputMessage(`{"bar": "123"}`),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent,
		response: response,
	})
	require.IsType(t, NextStepFinalOutput{}, result.NextStep)
	assert.Equal(t, Foo{Bar: "123"}, result.NextStep.(NextStepFinalOutput).Output)
}

func TestHandoffAndFinalOutputLeadsToHandoffNextStep(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:          "test_3",
		AgentHandoffs: []*Agent{agent1, agent2},
		OutputType:    OutputType[Foo](),
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFinalOutputMessage(`{"bar": "123"}`),
			getHandoffToolCall(agent1, "", ""),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent3,
		response: response,
	})

	require.IsType(t, NextStepHandoff{}, result.NextStep)
	assert.Same(t, agent1, result.NextStep.(NextStepHandoff).NewAgent)
}

func TestMultipleFinalOutputLeadsToFinalOutputNextStep(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	agent1 := &Agent{Name: "test_1"}
	agent2 := &Agent{Name: "test_2"}
	agent3 := &Agent{
		Name:          "test_3",
		AgentHandoffs: []*Agent{agent1, agent2},
		OutputType:    OutputType[Foo](),
	}
	response := ModelResponse{
		Output: []TResponseOutputItem{
			getFinalOutputMessage(`{"bar": "123"}`),
			getFinalOutputMessage(`{"bar": "456"}`),
		},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	result := getExecuteResult(t, getExecuteResultParams{
		agent:    agent3,
		response: response,
	})

	require.IsType(t, NextStepFinalOutput{}, result.NextStep)
	assert.Equal(t, Foo{Bar: "456"}, result.NextStep.(NextStepFinalOutput).Output)
}

func assertItemIsMessage(t *testing.T, item RunItem, agent *Agent, text string) {
	t.Helper()
	assert.Equal(t, MessageOutputItem{
		Agent: agent,
		RawItem: responses.ResponseOutputMessage{
			ID: "1",
			Content: []responses.ResponseOutputMessageContentUnion{
				{
					Text:        text,
					Type:        "output_text",
					Annotations: nil,
				},
			},
			Role:   constant.ValueOf[constant.Assistant](),
			Status: responses.ResponseOutputMessageStatusCompleted,
			Type:   constant.ValueOf[constant.Message](),
		},
		Type: "message_output_item",
	}, item)
}

func assertItemIsFunctionToolCall(t *testing.T, item RunItem, agent *Agent, name, arguments string) {
	t.Helper()
	assert.Equal(t, ToolCallItem{
		Agent: agent,
		RawItem: ResponseFunctionToolCall{
			ID:        "1",
			CallID:    "2",
			Name:      name,
			Arguments: arguments,
			Status:    "",
			Type:      constant.ValueOf[constant.FunctionCall](),
		},
		Type: "tool_call_item",
	}, item)
}

func assertItemIsFunctionToolCallOutput(t *testing.T, item RunItem, agent *Agent, output string) {
	t.Helper()
	assert.Equal(t, ToolCallOutputItem{
		Agent: agent,
		RawItem: ResponseInputItemFunctionCallOutputParam{
			ID:     param.Opt[string]{},
			CallID: "2",
			Output: output,
			Status: "",
			Type:   constant.ValueOf[constant.FunctionCallOutput](),
		},
		Output: output,
		Type:   "tool_call_output_item",
	}, item)
}

func getTextMessage(content string) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{ // responses.ResponseOutputMessage
		ID:   "1",
		Type: "message",
		Role: constant.ValueOf[constant.Assistant](),
		Content: []responses.ResponseOutputMessageContentUnion{{ // responses.ResponseOutputText
			Text:        content,
			Type:        "output_text",
			Annotations: nil,
		}},
		Status: string(responses.ResponseOutputMessageStatusCompleted),
	}
}

func getTextInputItem(content string) TResponseInputItem {
	return TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(content),
			},
			Role: responses.EasyInputMessageRoleUser,
		},
	}
}

func getFunctionToolCall(name string, arguments string) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{ // responses.ResponseFunctionToolCall
		ID:        "1",
		CallID:    "2",
		Type:      "function_call",
		Name:      name,
		Arguments: arguments,
	}
}

func getHandoffToolCall(toAgent *Agent, overrideName string, args string) responses.ResponseOutputItemUnion {
	name := overrideName
	if name == "" {
		name = DefaultHandoffToolName(toAgent)
	}
	return getFunctionToolCall(name, args)
}

func getFinalOutputMessage(args string) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{ // responses.ResponseOutputMessage
		ID:   "1",
		Type: "message",
		Role: constant.ValueOf[constant.Assistant](),
		Content: []responses.ResponseOutputMessageContentUnion{{ // responses.ResponseOutputText
			Text:        args,
			Type:        "output_text",
			Annotations: nil,
		}},
		Status: string(responses.ResponseOutputMessageStatusCompleted),
	}
}

type getExecuteResultParams struct {
	agent    *Agent
	response ModelResponse
	// optional
	originalInput  Input
	generatedItems []RunItem
	// optional
	hooks RunHooks
	// optional
	runConfig RunConfig
}

func getExecuteResult(t *testing.T, params getExecuteResultParams) SingleStepResult {
	handoffs, err := Runner{}.getHandoffs(t.Context(), params.agent)
	require.NoError(t, err)

	allTools, err := params.agent.GetAllTools(t.Context())
	require.NoError(t, err)

	processedResponse, err := RunImpl().ProcessModelResponse(
		t.Context(),
		params.agent,
		allTools,
		params.response,
		handoffs,
	)
	require.NoError(t, err)

	hooks := params.hooks
	if hooks == nil {
		hooks = NoOpRunHooks{}
	}

	input := params.originalInput
	if input == nil {
		input = InputString("hello")
	}

	result, err := RunImpl().ExecuteToolsAndSideEffects(
		t.Context(),
		params.agent,
		input,
		params.generatedItems,
		params.response,
		*processedResponse,
		params.agent.OutputType,
		hooks,
		params.runConfig,
	)
	require.NoError(t, err)
	return *result
}
