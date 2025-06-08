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
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/nlpodyssey/openai-agents-go/types/optional"
	"github.com/openai/openai-go/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleFirstRun(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("first"),
		},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("test"),
	})
	require.NoError(t, err)
	assert.Equal(t, agents.InputString("test"), result.Input)
	assert.Len(t, result.NewItems, 1)
	assert.Equal(t, "first", result.FinalOutput)
	require.Len(t, result.RawResponses, 1)
	assert.Equal(t, []agents.TResponseOutputItem{
		agentstesting.GetTextMessage("first"),
	}, result.RawResponses[0].Output)
	assert.Same(t, agent, result.LastAgent())

	assert.Len(t, result.ToInputList(), 2, "should have original input and generated item")

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("second"),
		},
	})

	result, err = agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input: agents.InputItems{
			agentstesting.GetTextInputItem("message"),
			agentstesting.GetTextInputItem("another_message"),
		},
	})
	require.NoError(t, err)
	assert.Len(t, result.NewItems, 1)
	assert.Equal(t, "second", result.FinalOutput)
	require.Len(t, result.RawResponses, 1)
	assert.Len(t, result.ToInputList(), 3, "should have original input and generated item")
}

func TestSubsequentRuns(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("third"),
		},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("test"),
	})
	require.NoError(t, err)
	assert.Equal(t, agents.InputString("test"), result.Input)
	assert.Len(t, result.NewItems, 1)
	assert.Len(t, result.ToInputList(), 2, "should have original input and generated item")

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("fourth"),
		},
	})

	result, err = agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputItems(result.ToInputList()),
	})
	require.NoError(t, err)
	assert.Len(t, result.Input.(agents.InputItems), 2)
	assert.Len(t, result.NewItems, 1)
	assert.Equal(t, "fourth", result.FinalOutput)
	require.Len(t, result.RawResponses, 1)
	assert.Equal(t, []agents.TResponseOutputItem{
		agentstesting.GetTextMessage("fourth"),
	}, result.RawResponses[0].Output)
	assert.Same(t, agent, result.LastAgent())
	assert.Len(t, result.ToInputList(), 3, "should have original input and generated items")
}

func TestToolCallRuns(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "tool_result"),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		// Second turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	assert.Equal(t, "done", result.FinalOutput)

	assert.Len(t, result.RawResponses, 2,
		"should have two responses: the first which produces a tool call, "+
			"and the second which handles the tool result")

	assert.Len(t, result.ToInputList(), 5,
		"should have five inputs: the original input, the message, "+
			"the tool call, the tool result, and the done message")
}

func TestHandoffs(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent3 := &agents.Agent{
		Name:     "agent_3",
		Model:    param.NewOpt(agents.NewAgentModel(model)),
		Handoffs: []agents.AgentHandoff{agent1, agent2},
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("some_function", "result"),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent3,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	assert.Equal(t, "done", result.FinalOutput)

	assert.Len(t, result.RawResponses, 3)
	assert.Len(t, result.ToInputList(), 7,
		"should have 7 inputs: orig input, tool call, tool result, "+
			"message, handoff, handoff result, and done message")
	assert.Same(t, agent1, result.LastAgent(), "should have handed off to agent1")
}

type AgentRunnerTestFoo struct {
	Bar string `json:"bar"`
}

type AgentRunnerTestFooSchema struct{}

func (AgentRunnerTestFooSchema) Name() string             { return "Foo" }
func (AgentRunnerTestFooSchema) IsPlainText() bool        { return false }
func (AgentRunnerTestFooSchema) IsStrictJSONSchema() bool { return true }
func (AgentRunnerTestFooSchema) JSONSchema() map[string]any {
	return map[string]any{
		"title":                "Foo",
		"type":                 "object",
		"required":             []string{"bar"},
		"additionalProperties": false,
		"properties": map[string]any{
			"bar": map[string]any{
				"title": "Bar",
				"type":  "string",
			},
		},
	}
}
func (AgentRunnerTestFooSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var v AgentRunnerTestFoo
	err := dec.Decode(&v)
	return v, err
}

func TestStructuredOutput(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("bar", "bar_result"),
		},
		OutputSchema: AgentRunnerTestFooSchema{},
	}
	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "foo_result"),
		},
		Handoffs: []agents.AgentHandoff{agent1},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("foo", `{"bar": "baz"}`),
		}},
		// Second turn: a message and a handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		// Third turn: tool call and structured output
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("bar", `{"bar": "baz"}`),
			agentstesting.GetFinalOutputMessage(`{"bar": "baz"}`),
		}},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent2,
		Input: agents.InputItems{
			agentstesting.GetTextInputItem("user_message"),
			agentstesting.GetTextInputItem("another_message"),
		},
	})
	require.NoError(t, err)
	assert.Equal(t, AgentRunnerTestFoo{Bar: "baz"}, result.FinalOutput)
	assert.Len(t, result.RawResponses, 3)
	assert.Len(t, result.ToInputList(), 10,
		"should have input: 2 orig inputs, function call, function call result, message, "+
			"handoff, handoff output, tool call, tool call result, final output message")
	assert.Same(t, agent1, result.LastAgent(), "should have handed off to agent1")
}

func RemoveNewItems(_ context.Context, handoffInputData agents.HandoffInputData) (agents.HandoffInputData, error) {
	return agents.HandoffInputData{
		InputHistory:    handoffInputData.InputHistory,
		PreHandoffItems: nil,
		NewItems:        nil,
	}, nil
}

func TestHandoffFilters(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Handoffs: []agents.AgentHandoff{
			agents.UnsafeHandoffFromAgent(agents.HandoffFromAgentParams{
				Agent:       agent1,
				InputFilter: RemoveNewItems,
			}),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("1"),
			agentstesting.GetTextMessage("2"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("last"),
		}},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent2,
		Input:         agents.InputString("user_message"),
	})

	require.NoError(t, err)
	assert.Equal(t, "last", result.FinalOutput)
	assert.Len(t, result.RawResponses, 2)
	assert.Len(t, result.ToInputList(), 2, "should only have 2 inputs: orig input and last message")
}

func TestInputFilterError(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	onInvokeHandoff := func(context.Context, *runcontext.RunContextWrapper, string) (*agents.Agent, error) {
		return agent1, nil
	}

	inputFilterError := errors.New("input filter error")
	invalidInputFilter := func(context.Context, agents.HandoffInputData) (agents.HandoffInputData, error) {
		return agents.HandoffInputData{}, inputFilterError
	}

	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Handoffs: []agents.AgentHandoff{
			agents.Handoff{
				ToolName:        agents.DefaultHandoffToolName(agent1),
				ToolDescription: agents.DefaultHandoffToolDescription(agent1),
				InputJSONSchema: map[string]any{},
				OnInvokeHandoff: onInvokeHandoff,
				AgentName:       agent1.Name,
				InputFilter:     invalidInputFilter,
			},
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("1"),
			agentstesting.GetTextMessage("2"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("last"),
		}},
	})

	_, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent2,
		Input:         agents.InputString("user_message"),
	})
	assert.ErrorIs(t, err, inputFilterError)
}

func TestHandoffOnInput(t *testing.T) {
	callOutput := optional.None[string]()

	onInput := func(_ context.Context, _ *runcontext.RunContextWrapper, jsonInput any) error {
		r := strings.NewReader(jsonInput.(string))
		dec := json.NewDecoder(r)
		dec.DisallowUnknownFields()
		var v AgentRunnerTestFoo
		err := dec.Decode(&v)
		if err != nil {
			return err
		}
		callOutput = optional.Value(v.Bar)
		return nil
	}

	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Handoffs: []agents.AgentHandoff{
			agents.UnsafeHandoffFromAgent(agents.HandoffFromAgentParams{
				Agent:           agent1,
				OnHandoff:       agents.OnHandoffWithInput(onInput),
				InputJSONSchema: AgentRunnerTestFooSchema{}.JSONSchema(),
			}),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("1"),
			agentstesting.GetTextMessage("2"),
			agentstesting.GetHandoffToolCall(agent1, "", `{"bar": "test_input"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("last"),
		}},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent2,
		Input:         agents.InputString("user_message"),
	})
	assert.NoError(t, err)
	assert.Equal(t, "last", result.FinalOutput)
	assert.Equal(t, optional.Value("test_input"), callOutput)
}

func TestHandoffOnInputError(t *testing.T) {
	onInputError := errors.New("on input error")
	onInput := func(context.Context, *runcontext.RunContextWrapper, any) error {
		return onInputError
	}

	model := agentstesting.NewFakeModel(nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Handoffs: []agents.AgentHandoff{
			agents.UnsafeHandoffFromAgent(agents.HandoffFromAgentParams{
				Agent:           agent1,
				OnHandoff:       agents.OnHandoffWithInput(onInput),
				InputJSONSchema: AgentRunnerTestFooSchema{}.JSONSchema(),
			}),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("1"),
			agentstesting.GetTextMessage("2"),
			agentstesting.GetHandoffToolCall(agent1, "", `{"bar": "test_input"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("last"),
		}},
	})

	_, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent2,
		Input:         agents.InputString("user_message"),
	})
	assert.Error(t, err, onInputError)
}

func TestInvalidHandoffInputJSONCausesError(t *testing.T) {
	agent := &agents.Agent{Name: "test"}
	h := agents.UnsafeHandoffFromAgent(agents.HandoffFromAgentParams{
		Agent: agent,
		OnHandoff: agents.OnHandoffWithInput(
			func(context.Context, *runcontext.RunContextWrapper, any) error { return nil },
		),
		InputJSONSchema: AgentRunnerTestFooSchema{}.JSONSchema(),
	})

	cw := runcontext.NewRunContextWrapper(nil)
	var target agents.ModelBehaviorError

	_, err := h.OnInvokeHandoff(t.Context(), cw, "")
	assert.ErrorAs(t, err, &target)

	_, err = h.OnInvokeHandoff(t.Context(), cw, `{"foo": "y"}`)
	assert.ErrorAs(t, err, &target)
}

func TestInputGuardrailTripwireTriggeredCausesError(t *testing.T) {
	guardrailFunction := func(
		context.Context, *runcontext.RunContextWrapper, *agents.Agent, agents.Input,
	) (agents.GuardrailFunctionOutput, error) {
		return agents.GuardrailFunctionOutput{
			OutputInfo:        nil,
			TripwireTriggered: true,
		}, nil
	}

	model := agentstesting.NewFakeModel(&agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("user_message"),
		},
	})

	agent := &agents.Agent{
		Name: "test",
		InputGuardrails: []agents.InputGuardrail{{
			Name:              "guardrail_function",
			GuardrailFunction: guardrailFunction,
		}},
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	_, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("user_message"),
	})
	var target agents.InputGuardrailTripwireTriggeredError
	assert.ErrorAs(t, err, &target)
}

func TestOutputGuardrailTripwireTriggeredCausesError(t *testing.T) {
	guardrailFunction := func(
		context.Context, *runcontext.RunContextWrapper, *agents.Agent, any,
	) (agents.GuardrailFunctionOutput, error) {
		return agents.GuardrailFunctionOutput{
			OutputInfo:        nil,
			TripwireTriggered: true,
		}, nil
	}

	model := agentstesting.NewFakeModel(&agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("user_message"),
		},
	})

	agent := &agents.Agent{
		Name: "test",
		OutputGuardrails: []agents.OutputGuardrail{{
			Name:              "guardrail_function",
			GuardrailFunction: guardrailFunction,
		}},
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	_, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("user_message"),
	})
	var target agents.OutputGuardrailTripwireTriggeredError
	assert.ErrorAs(t, err, &target)
}

var TestToolOne = agents.FunctionTool{
	Name:        "test_tool_one",
	Description: "",
	ParamsJSONSchema: map[string]any{
		"title":                "test_tool_one_args",
		"type":                 "object",
		"required":             []string{},
		"additionalProperties": false,
		"properties":           map[string]any{},
	},
	OnInvokeTool: func(context.Context, *runcontext.RunContextWrapper, string) (any, error) {
		return AgentRunnerTestFoo{Bar: "tool_one_result"}, nil
	},
	StrictJSONSchema: param.NewOpt(true),
}

var TestToolTwo = agents.FunctionTool{
	Name:        "test_tool_two",
	Description: "",
	ParamsJSONSchema: map[string]any{
		"title":                "test_tool_two_args",
		"type":                 "object",
		"required":             []string{},
		"additionalProperties": false,
		"properties":           map[string]any{},
	},
	OnInvokeTool: func(context.Context, *runcontext.RunContextWrapper, string) (any, error) {
		return AgentRunnerTestFoo{Bar: "tool_two_result"}, nil
	},
	StrictJSONSchema: param.NewOpt(true),
}

func TestToolUseBehaviorFirstOutput(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "tool_result"),
			TestToolOne,
			TestToolTwo,
		},
		ToolUseBehavior: agents.StopOnFirstTool{},
		OutputSchema:    AgentRunnerTestFooSchema{},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("test_tool_one", ""),
			agentstesting.GetFunctionToolCall("test_tool_two", ""),
		}},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	assert.Equal(t, AgentRunnerTestFoo{Bar: "tool_one_result"}, result.FinalOutput)
}

var CustomToolUseBehavior = func(_ *runcontext.RunContextWrapper, results []agents.FunctionToolResult) (agents.ToolsToFinalOutputResult, error) {
	if slices.ContainsFunc(results, func(r agents.FunctionToolResult) bool { return r.Tool.Name == "test_tool_one" }) {
		return agents.ToolsToFinalOutputResult{
			IsFinalOutput: true,
			FinalOutput:   param.NewOpt[any]("the_final_output"),
		}, nil
	}
	return agents.ToolsToFinalOutputResult{IsFinalOutput: false}, nil
}

func TestToolUseBehaviorCustomFunction(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "tool_result"),
			TestToolOne,
			TestToolTwo,
		},
		ToolUseBehavior: agents.ToolsToFinalOutputFunction(CustomToolUseBehavior),
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("test_tool_two", ""),
		}},
		// Second turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("test_tool_one", ""),
			agentstesting.GetFunctionToolCall("test_tool_two", ""),
		}},
	})

	result, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("user_message"),
	})
	require.NoError(t, err)
	assert.Len(t, result.RawResponses, 2)
	assert.Equal(t, "the_final_output", result.FinalOutput)
}

func TestModelSettingsOverride(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		ModelSettings: modelsettings.ModelSettings{
			Temperature: param.NewOpt(1.0),
			MaxTokens:   param.NewOpt[int64](1000),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
		}},
	})

	_, err := agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("user_message"),
		RunConfig: agents.RunConfig{
			ModelSettings: optional.Value(modelsettings.ModelSettings{
				Temperature: param.NewOpt(0.5),
			}),
		},
	})
	require.NoError(t, err)

	// Temperature is overridden by Runner.run, but MaxTokens is not
	assert.Equal(t, param.NewOpt(0.5), model.LastTurnArgs.ModelSettings.Temperature)
	assert.Equal(t, param.NewOpt[int64](1000), model.LastTurnArgs.ModelSettings.MaxTokens)
}

func TestPreviousResponseIDPassedBetweenRuns(t *testing.T) {
	// Test that PreviousResponseID is passed to the model on subsequent runs.

	model := agentstesting.NewFakeModel(&agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		},
	})
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	assert.Equal(t, "", model.LastTurnArgs.PreviousResponseID)
	_, _ = agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent:      agent,
		Input:              agents.InputString("test"),
		PreviousResponseID: "resp-non-streamed-test",
	})
	assert.Equal(t, "resp-non-streamed-test", model.LastTurnArgs.PreviousResponseID)
}

func TestMultiTurnPreviousResponseIDPassedBetweenRuns(t *testing.T) {
	// Test that PreviousResponseID is passed to the model on subsequent runs.

	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "foo_result"),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		// Second turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	assert.Equal(t, "", model.LastTurnArgs.PreviousResponseID)
	_, _ = agents.Runner().Run(t.Context(), agents.RunParams{
		StartingAgent:      agent,
		Input:              agents.InputString("test"),
		PreviousResponseID: "resp-test-123",
	})
	assert.Equal(t, "resp-test-123", model.LastTurnArgs.PreviousResponseID)
}

func TestPreviousResponseIDPassedBetweenRunsStreamed(t *testing.T) {
	// Test that PreviousResponseID is passed to the model on subsequent streamed runs.

	model := agentstesting.NewFakeModel(&agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		},
	})
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	assert.Equal(t, "", model.LastTurnArgs.PreviousResponseID)

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent:      agent,
		Input:              agents.InputString("test"),
		PreviousResponseID: "resp-stream-test",
	})
	require.NoError(t, err)
	err = result.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "resp-stream-test", model.LastTurnArgs.PreviousResponseID)
}

func TestPreviousResponseIDPassedBetweenRunsStreamedMultiTurn(t *testing.T) {
	// Test that PreviousResponseID is passed to the model on subsequent streamed runs.

	model := agentstesting.NewFakeModel(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "foo_result"),
		},
	}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		// Second turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	assert.Equal(t, "", model.LastTurnArgs.PreviousResponseID)

	result, err := agents.Runner().RunStreamed(t.Context(), agents.RunStreamedParams{
		StartingAgent:      agent,
		Input:              agents.InputString("test"),
		PreviousResponseID: "resp-stream-test",
	})
	require.NoError(t, err)
	err = result.StreamEvents(func(event agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "resp-stream-test", model.LastTurnArgs.PreviousResponseID)
}
