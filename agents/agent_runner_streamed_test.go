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
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleFirstRunStreamed(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("first"),
		},
	})

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "test")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, agents.InputString("test"), result.Input())
	assert.Len(t, result.NewItems(), 1)
	assert.Equal(t, "first", result.FinalOutput())
	require.Len(t, result.RawResponses(), 1)
	assert.Equal(t, []agents.TResponseOutputItem{
		agentstesting.GetTextMessage("first"),
	}, result.RawResponses()[0].Output)
	assert.Same(t, agent, result.LastAgent())
	assert.Len(t, result.ToInputList(), 2, "should have original input and generated item")

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("second"),
		},
	})

	result, err = agents.Runner{}.RunInputsStreamed(t.Context(), agent, []agents.TResponseInputItem{
		agentstesting.GetTextInputItem("message"),
		agentstesting.GetTextInputItem("another_message"),
	})
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Len(t, result.NewItems(), 1)
	assert.Equal(t, "second", result.FinalOutput())
	assert.Len(t, result.RawResponses(), 1)
	assert.Len(t, result.ToInputList(), 3, "should have original input and generated item")
}

func TestSubsequentRunsStreamed(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("third"),
		},
	})

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "test")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, agents.InputString("test"), result.Input())
	assert.Len(t, result.NewItems(), 1)
	assert.Len(t, result.ToInputList(), 2, "should have original input and generated item")

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("fourth"),
		},
	})

	result, err = agents.Runner{}.RunInputsStreamed(t.Context(), agent, result.ToInputList())
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Len(t, result.Input().(agents.InputItems), 2)
	assert.Len(t, result.NewItems(), 1)
	assert.Equal(t, "fourth", result.FinalOutput())
	require.Len(t, result.RawResponses(), 1)
	assert.Equal(t, []agents.TResponseOutputItem{
		agentstesting.GetTextMessage("fourth"),
	}, result.RawResponses()[0].Output)
	assert.Same(t, agent, result.LastAgent())
	assert.Len(t, result.ToInputList(), 3, "should have original input and generated items")
}

func TestToolCallRunsStreamed(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
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

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "done", result.FinalOutput())

	assert.Len(t, result.RawResponses(), 2,
		"should have two responses: the first which produces a tool call, "+
			"and the second which handles the tool result")

	assert.Len(t, result.ToInputList(), 5,
		"should have five inputs: the original input, the message, "+
			"the tool call, the tool result, and the done message")
}

func TestHandoffsStreaming(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent3 := &agents.Agent{
		Name:          "agent_3",
		Model:         param.NewOpt(agents.NewAgentModel(model)),
		AgentHandoffs: []*agents.Agent{agent1, agent2},
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

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "done", result.FinalOutput())

	assert.Len(t, result.RawResponses(), 3)
	assert.Len(t, result.ToInputList(), 7,
		"should have 7 inputs: orig input, tool call, tool result, "+
			"message, handoff, handoff result, and done message")
	assert.Same(t, agent1, result.LastAgent(), "should have handed off to agent1")
}

func TestStructuredOutputStreamed(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("bar", "bar_result"),
		},
		OutputType: agents.OutputType[AgentRunnerTestFoo](),
	}
	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "foo_result"),
		},
		AgentHandoffs: []*agents.Agent{agent1},
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

	result, err := agents.Runner{}.RunInputsStreamed(t.Context(), agent2, []agents.TResponseInputItem{
		agentstesting.GetTextInputItem("user_message"),
		agentstesting.GetTextInputItem("another_message"),
	})
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, AgentRunnerTestFoo{Bar: "baz"}, result.FinalOutput())
	assert.Len(t, result.RawResponses(), 3)
	assert.Len(t, result.ToInputList(), 10,
		"should have input: 2 orig inputs, function call, function call result, message, "+
			"handoff, handoff output, tool call, tool call result, final output message")
	assert.Same(t, agent1, result.LastAgent(), "should have handed off to agent1")
}

func TestHandoffFiltersStreamed(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}
	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Handoffs: []agents.Handoff{
			agents.HandoffFromAgent(agents.HandoffFromAgentParams{
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

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent2, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "last", result.FinalOutput())
	assert.Len(t, result.RawResponses(), 2)
	assert.Len(t, result.ToInputList(), 2, "should only have 2 inputs: orig input and last message")
}

func TestInputFilterErrorStreamed(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	onInvokeHandoff := func(context.Context, string) (*agents.Agent, error) {
		return agent1, nil
	}

	inputFilterError := errors.New("input filter error")
	invalidInputFilter := func(context.Context, agents.HandoffInputData) (agents.HandoffInputData, error) {
		return agents.HandoffInputData{}, inputFilterError
	}

	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Handoffs: []agents.Handoff{
			{
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

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent2, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	assert.ErrorIs(t, err, inputFilterError)
}

func TestHandoffOnInputStreamed(t *testing.T) {
	callOutput := ""

	onInput := func(_ context.Context, jsonInput any) error {
		r := strings.NewReader(jsonInput.(string))
		dec := json.NewDecoder(r)
		dec.DisallowUnknownFields()
		var v AgentRunnerTestFoo
		err := dec.Decode(&v)
		if err != nil {
			return err
		}
		callOutput = v.Bar
		return nil
	}

	model := agentstesting.NewFakeModel(false, nil)
	agent1 := &agents.Agent{
		Name:  "agent_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	schema, err := agents.OutputType[AgentRunnerTestFoo]().JSONSchema()
	require.NoError(t, err)

	agent2 := &agents.Agent{
		Name:  "agent_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Handoffs: []agents.Handoff{
			agents.HandoffFromAgent(agents.HandoffFromAgentParams{
				Agent:           agent1,
				OnHandoff:       agents.OnHandoffWithInput(onInput),
				InputJSONSchema: schema,
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

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent2, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "last", result.FinalOutput())
	assert.Equal(t, "test_input", callOutput)
}

func TestInputGuardrailTripwireTriggeredCausesErrorStreamed(t *testing.T) {
	guardrailFunction := func(
		context.Context, *agents.Agent, agents.Input,
	) (agents.GuardrailFunctionOutput, error) {
		return agents.GuardrailFunctionOutput{
			OutputInfo:        nil,
			TripwireTriggered: true,
		}, nil
	}

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
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

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	assert.ErrorAs(t, err, &agents.InputGuardrailTripwireTriggeredError{})
}

func TestOutputGuardrailTripwireTriggeredCausesErrorStreamed(t *testing.T) {
	guardrailFunction := func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
		return agents.GuardrailFunctionOutput{
			OutputInfo:        nil,
			TripwireTriggered: true,
		}, nil
	}

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("first_test"),
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

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	assert.ErrorAs(t, err, &agents.OutputGuardrailTripwireTriggeredError{})
}

func TestRunInputGuardrailTripwireTriggeredCausesErrorStreamed(t *testing.T) {
	guardrailFunction := func(context.Context, *agents.Agent, agents.Input) (agents.GuardrailFunctionOutput, error) {
		return agents.GuardrailFunctionOutput{
			OutputInfo:        nil,
			TripwireTriggered: true,
		}, nil
	}

	model := agentstesting.NewFakeModel(false, nil)

	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	result, err := (agents.Runner{Config: agents.RunConfig{
		InputGuardrails: []agents.InputGuardrail{{
			Name:              "guardrail_function",
			GuardrailFunction: guardrailFunction,
		}},
	}}).RunStreamed(t.Context(), agent, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	assert.ErrorAs(t, err, &agents.InputGuardrailTripwireTriggeredError{})
}

func TestRunOutputGuardrailTripwireTriggeredCausesErrorStreamed(t *testing.T) {
	guardrailFunction := func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
		return agents.GuardrailFunctionOutput{
			OutputInfo:        nil,
			TripwireTriggered: true,
		}, nil
	}

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("first_test"),
		},
	})

	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(model)),
	}

	result, err := (agents.Runner{Config: agents.RunConfig{
		OutputGuardrails: []agents.OutputGuardrail{{
			Name:              "guardrail_function",
			GuardrailFunction: guardrailFunction,
		}},
	}}).RunStreamed(t.Context(), agent, "user_message")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	assert.ErrorAs(t, err, &agents.OutputGuardrailTripwireTriggeredError{})
}

func TestStreamingEvents(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	agent1 := &agents.Agent{
		Name:  "test_1",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("bar", "bar_result"),
		},
		OutputType: agents.OutputType[AgentRunnerTestFoo](),
	}

	agent2 := &agents.Agent{
		Name:  "test_2",
		Model: param.NewOpt(agents.NewAgentModel(model)),
		Tools: []agents.Tool{
			agentstesting.GetFunctionTool("foo", "foo_result"),
		},
		AgentHandoffs: []*agents.Agent{agent1},
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

	eventCounts := make(map[string]int)
	var itemData []agents.RunItem
	var agentData []agents.AgentUpdatedStreamEvent

	result, err := agents.Runner{}.RunInputsStreamed(t.Context(), agent2, []agents.TResponseInputItem{
		agentstesting.GetTextInputItem("user_message"),
		agentstesting.GetTextInputItem("another_message"),
	})
	require.NoError(t, err)

	err = result.StreamEvents(func(event agents.StreamEvent) error {
		switch e := event.(type) {
		case agents.RawResponsesStreamEvent:
			eventCounts[e.Type] += 1
		case agents.RunItemStreamEvent:
			eventCounts[e.Type] += 1
			itemData = append(itemData, e.Item)
		case agents.AgentUpdatedStreamEvent:
			eventCounts[e.Type] += 1
			agentData = append(agentData, e)
		default:
			t.Fatalf("unexpected StreamEvent type %T", e)
		}
		return nil
	})
	require.NoError(t, err)

	assert.Equal(t, AgentRunnerTestFoo{Bar: "baz"}, result.FinalOutput())
	assert.Len(t, result.RawResponses(), 3)
	assert.Len(t, result.ToInputList(), 10,
		"should have input: 2 orig inputs, function call, function call result, "+
			"message, handoff, handoff output, tool call, tool call result, final output")
	assert.Same(t, agent1, result.LastAgent(), "should have handed off to agent1")

	// Now let's check the events

	expectedItemTypeMap := map[string]int{
		"tool_call":        2,
		"tool_call_output": 2,
		"message":          2,
		"handoff":          1,
		"handoff_output":   1,
	}

	totalExpectedItemCount := 0
	for _, n := range expectedItemTypeMap {
		totalExpectedItemCount += n
	}

	assert.Equalf(t, totalExpectedItemCount, eventCounts["run_item_stream_event"],
		"Expected %d events, got %d. Expected events were: %v, got %v",
		totalExpectedItemCount, eventCounts["run_item_stream_event"],
		expectedItemTypeMap, eventCounts)

	assert.Len(t, itemData, totalExpectedItemCount)
	require.Len(t, agentData, 2)
	assert.Same(t, agent2, agentData[0].NewAgent)
	assert.Same(t, agent1, agentData[1].NewAgent)
}

func TestDynamicToolAdditionRunStreamed(t *testing.T) {
	// Test that tools can be added to an agent during a run.
	model := agentstesting.NewFakeModel(false, nil)

	agent := agents.New("test").
		WithModelInstance(model).
		WithToolUseBehavior(agents.RunLLMAgain())

	tool2Called := false
	tool2 := agents.NewFunctionTool("tool2", "", func(ctx context.Context, args struct{}) (string, error) {
		tool2Called = true
		return "result2", nil
	})

	addTool := agents.NewFunctionTool("add_tool", "", func(ctx context.Context, args struct{}) (string, error) {
		agent.AddTool(tool2)
		return "added", nil
	})

	agent.AddTool(addTool)

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("add_tool", `{}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("tool2", `{}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	result, err := agents.Runner{}.RunStreamed(t.Context(), agent, "test")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	assert.Equal(t, "done", result.FinalOutput())
	assert.True(t, tool2Called)
}
