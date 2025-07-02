package agents_test

import (
	"context"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/tracing/tracingtesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type TestingError struct {
	Message string
}

func (err TestingError) Error() string { return err.Message }

func TestSingleTurnModelError(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	testError := TestingError{Message: "test error"}

	model := agentstesting.NewFakeModel(true, &agentstesting.FakeModelTurnOutput{
		Error: testError,
	})

	agent := agents.New("test_agent").WithModelInstance(model)

	_, err := agents.Run(t.Context(), agent, "first_test")
	assert.ErrorIs(t, err, testError)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
					"children": []m{
						{
							"type": "generation",
							"error": m{
								"message": "Error",
								"data":    m{"name": "agents_test.TestingError", "message": "test error"},
							},
						},
					},
				},
			},
		},
	}, spans)
}

func TestMultiTurnNoHandoffs(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(true, nil)

	agent := agents.New("test_agent").
		WithModelInstance(model).
		WithTools(agentstesting.GetFunctionTool("foo", "tool_result"))

	testError := TestingError{Message: "test error"}

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a message and tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		// Second turn: error
		{Error: testError},
		// Third turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	_, err := agents.Run(t.Context(), agent, "first_test")
	assert.ErrorIs(t, err, testError)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent",
						"handoffs":    []string{},
						"tools":       []string{"foo"},
						"output_type": "string",
					},
					"children": []m{
						{"type": "generation"},
						{
							"type": "function",
							"data": m{
								"name":   "foo",
								"input":  `{"a": "b"}`,
								"output": "tool_result",
							},
						},
						{
							"type": "generation",
							"error": m{
								"message": "Error",
								"data":    m{"name": "agents_test.TestingError", "message": "test error"},
							},
						},
					},
				},
			},
		},
	}, spans)
}

func TestToolCallError(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(true, nil)

	testError := agents.NewModelBehaviorError("test error")

	agent := agents.New("test_agent").
		WithModelInstance(model).
		WithTools(agentstesting.GetFunctionToolErr("foo", testError))

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetFunctionToolCall("foo", "bad_json"),
		},
	})

	_, err := agents.Run(t.Context(), agent, "first_test")
	assert.ErrorIs(t, err, testError)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent",
						"handoffs":    []string{},
						"tools":       []string{"foo"},
						"output_type": "string",
					},
					"children": []m{
						{"type": "generation"},
						{
							"type": "function",
							"error": m{
								"message": "Error running tool",
								"data": m{
									"tool_name": "foo",
									"error":     "error running tool foo: test error",
								},
							},
							"data": m{"name": "foo", "input": "bad_json", "output": ""},
						},
					},
				},
			},
		},
	}, spans)
}

func TestMultipleHandoffDoesNotError(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(true, nil)

	agent1 := agents.New("agent_1").WithModelInstance(model)
	agent2 := agents.New("agent_2").WithModelInstance(model)
	agent3 := agents.New("agent_3").
		WithModelInstance(model).
		WithAgentHandoffs(agent1, agent2).
		WithTools(agentstesting.GetFunctionTool("some_function", "result"))

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Second turn: a message and 2 handoffs
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
			agentstesting.GetHandoffToolCall(agent2, "", ""),
		}},
		// Third turn: text message
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	result, err := agents.Run(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	assert.Same(t, agent1, result.LastAgent, "should have picked first handoff")

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "agent_3",
						"handoffs":    []string{"agent_1", "agent_2"},
						"tools":       []string{"some_function"},
						"output_type": "string",
					},
					"children": []m{
						{"type": "generation"},
						{
							"type": "function",
							"data": m{
								"name":   "some_function",
								"input":  `{"a": "b"}`,
								"output": "result",
							},
						},
						{"type": "generation"},
						{
							"type": "handoff",
							"data": m{"from_agent": "agent_3", "to_agent": "agent_1"},
							"error": m{
								"data": m{
									"requested_agents": []string{"agent_1", "agent_2"},
								},
								"message": "Multiple handoffs requested",
							},
						},
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
					"children": []m{{"type": "generation"}},
				},
			},
		},
	}, spans)
}

func TestMultipleFinalOutputDoesNotError(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	type Foo struct {
		Bar string `json:"bar"`
	}

	model := agentstesting.NewFakeModel(true, nil)

	agent := agents.New("test").
		WithModelInstance(model).
		WithOutputType(agents.OutputType[Foo]())

	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetFinalOutputMessage(`{"bar": "baz"}`),
			agentstesting.GetFinalOutputMessage(`{"bar": "abc"}`),
		},
	})

	result, err := agents.Run(t.Context(), agent, "first_test")
	require.NoError(t, err)
	assert.Equal(t, Foo{Bar: "abc"}, result.FinalOutput)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "agents_test.Foo",
					},
					"children": []m{{"type": "generation"}},
				},
			},
		},
	}, spans)
}

func TestHandoffsLeadToCorrectAgentSpans(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(true, nil)

	agent1 := agents.New("test_agent_1").
		WithModelInstance(model).
		WithTools(agentstesting.GetFunctionTool("some_function", "result"))
	agent2 := agents.New("test_agent_2").
		WithModelInstance(model).
		WithAgentHandoffs(agent1).
		WithTools(agentstesting.GetFunctionTool("some_function", "result"))
	agent3 := agents.New("test_agent_3").
		WithModelInstance(model).
		WithAgentHandoffs(agent1, agent2).
		WithTools(agentstesting.GetFunctionTool("some_function", "result"))

	agent1.AgentHandoffs = append(agent1.AgentHandoffs, agent3)

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		// First turn: a tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		//  Second turn: a message and 2 handoffs
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("a_message"),
			agentstesting.GetHandoffToolCall(agent1, "", ""),
			agentstesting.GetHandoffToolCall(agent2, "", ""),
		}},
		// Third turn: tool call
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetFunctionToolCall("some_function", `{"a": "b"}`),
		}},
		// Fourth turn: handoff
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetHandoffToolCall(agent3, "", ""),
		}},
		// Fifth turn: text message
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("done")}},
	})

	result, err := agents.Run(t.Context(), agent3, "user_message")
	require.NoError(t, err)
	assert.Same(t, agent3, result.LastAgent)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_3",
						"handoffs":    []string{"test_agent_1", "test_agent_2"},
						"tools":       []string{"some_function"},
						"output_type": "string",
					},
					"children": []m{
						{"type": "generation"},
						{
							"type": "function",
							"data": m{
								"name":   "some_function",
								"input":  `{"a": "b"}`,
								"output": "result",
							},
						},
						{"type": "generation"},
						{
							"type": "handoff",
							"data": m{"from_agent": "test_agent_3", "to_agent": "test_agent_1"},
							"error": m{
								"data": m{
									"requested_agents": []string{"test_agent_1", "test_agent_2"},
								},
								"message": "Multiple handoffs requested",
							},
						},
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{"test_agent_3"},
						"tools":       []string{"some_function"},
						"output_type": "string",
					},
					"children": []m{
						{"type": "generation"},
						{
							"type": "function",
							"data": m{
								"name":   "some_function",
								"input":  `{"a": "b"}`,
								"output": "result",
							},
						},
						{"type": "generation"},
						{
							"type": "handoff",
							"data": m{"from_agent": "test_agent_1", "to_agent": "test_agent_3"},
						},
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_3",
						"handoffs":    []string{"test_agent_1", "test_agent_2"},
						"tools":       []string{"some_function"},
						"output_type": "string",
					},
					"children": []m{{"type": "generation"}},
				},
			},
		},
	}, spans)
}

func TestMaxTurnsExceeded(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	type Foo struct {
		Bar string `json:"bar"`
	}

	model := agentstesting.NewFakeModel(true, nil)

	agent := agents.New("test").
		WithModelInstance(model).
		WithOutputType(agents.OutputType[Foo]()).
		WithTools(agentstesting.GetFunctionTool("foo", "result"))

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("foo", "")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("foo", "")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("foo", "")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("foo", "")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetFunctionToolCall("foo", "")}},
	})

	runner := agents.Runner{Config: agents.RunConfig{MaxTurns: 2}}
	_, err := runner.Run(t.Context(), agent, "first_test")
	assert.ErrorAs(t, err, &agents.MaxTurnsExceededError{})

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type":  "agent",
					"error": m{"message": "Max turns exceeded", "data": m{"max_turns": uint64(2)}},
					"data": m{
						"name":        "test",
						"handoffs":    []string{},
						"tools":       []string{"foo"},
						"output_type": "agents_test.Foo",
					},
					"children": []m{
						{"type": "generation"},
						{
							"type": "function",
							"data": m{"name": "foo", "output": "result"},
						},
						{"type": "generation"},
						{
							"type": "function",
							"data": m{"name": "foo", "output": "result"},
						},
					},
				},
			},
		},
	}, spans)
}

func TestGuardrailError(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("some_message")},
	})

	agent := agents.New("test").
		WithModelInstance(model).
		WithInputGuardrails([]agents.InputGuardrail{
			{
				Name: "input_guardrail_function",
				GuardrailFunction: func(context.Context, *agents.Agent, agents.Input) (agents.GuardrailFunctionOutput, error) {
					return agents.GuardrailFunctionOutput{
						OutputInfo:        nil,
						TripwireTriggered: true,
					}, nil
				},
			},
		})

	_, err := agents.Run(t.Context(), agent, "user_message")
	assert.ErrorAs(t, err, &agents.InputGuardrailTripwireTriggeredError{})

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"error": m{
						"message": "Guardrail tripwire triggered",
						"data":    m{"guardrail": "input_guardrail_function"},
					},
					"data": m{
						"name":        "test",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
					"children": []m{
						{
							"type": "guardrail",
							"data": m{"name": "input_guardrail_function", "triggered": true},
						},
					},
				},
			},
		},
	}, spans)
}

func TestOutputGuardrailError(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("some_message")},
	})

	agent := agents.New("test").
		WithModelInstance(model).
		WithOutputGuardrails([]agents.OutputGuardrail{
			{
				Name: "output_guardrail_function",
				GuardrailFunction: func(context.Context, *agents.Agent, any) (agents.GuardrailFunctionOutput, error) {
					return agents.GuardrailFunctionOutput{
						OutputInfo:        nil,
						TripwireTriggered: true,
					}, nil
				},
			},
		})

	_, err := agents.Run(t.Context(), agent, "user_message")
	assert.ErrorAs(t, err, &agents.OutputGuardrailTripwireTriggeredError{})

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"error": m{
						"message": "Guardrail tripwire triggered",
						"data":    m{"guardrail": "output_guardrail_function"},
					},
					"data": m{
						"name":        "test",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
					"children": []m{
						{
							"type": "guardrail",
							"data": m{"name": "output_guardrail_function", "triggered": true},
						},
					},
				},
			},
		},
	}, spans)
}
