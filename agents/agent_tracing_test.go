package agents_test

import (
	"context"
	"testing"
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/nlpodyssey/openai-agents-go/tracing/tracingtesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSingleRunIsSingleTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	agent := agents.New("test_agent").WithModelInstance(
		agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")},
		}),
	)

	_, err := agents.Run(t.Context(), agent, "first_test")
	require.NoError(t, err)

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
				},
			},
		},
	}, spans)
}

func TestMultipleRunsAreMultipleTraces(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("second_test")}},
	})
	agent := agents.New("test_agent_1").WithModelInstance(model)

	_, err := agents.Run(t.Context(), agent, "first_test")
	require.NoError(t, err)
	_, err = agents.Run(t.Context(), agent, "second_test")
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
			},
		},
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
			},
		},
	}, spans)
}

func TestWrappedTraceIsSingleTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("second_test")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("third_test")}},
	})

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test_workflow"},
		func(ctx context.Context, _ tracing.Trace) error {
			agent := agents.New("test_agent_1").WithModelInstance(model)
			for _, input := range []string{"first_test", "second_test", "third_test"} {
				_, err := agents.Run(ctx, agent, input)
				if err != nil {
					return err
				}
			}
			return nil
		})
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test_workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
			},
		},
	}, spans)
}

func TestParentDisabledTraceDisabledAgentTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test_workflow", Disabled: true},
		func(ctx context.Context, _ tracing.Trace) error {
			agent := agents.New("test_agent").WithModelInstance(
				agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")},
				}),
			)

			_, err := agents.Run(ctx, agent, "first_test")
			return err
		})
	require.NoError(t, err)

	tracingtesting.RequireNoTraces(t)
}

func TestManualDisablingWorks(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	agent := agents.New("test_agent").WithModelInstance(
		agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")},
		}),
	)

	runner := agents.Runner{
		Config: agents.RunConfig{
			TracingDisabled: true,
		},
	}
	_, err := runner.Run(t.Context(), agent, "first_test")
	require.NoError(t, err)

	tracingtesting.RequireNoTraces(t)
}

func TestTraceConfigWorks(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	agent := agents.New("test_agent").WithModelInstance(
		agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")},
		}),
	)

	runner := agents.Runner{
		Config: agents.RunConfig{
			WorkflowName: "Foo bar",
			GroupID:      "123",
			TraceID:      "trace_456",
		},
	}
	_, err := runner.Run(t.Context(), agent, "first_test")
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, true, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"id":            "trace_456",
			"workflow_name": "Foo bar",
			"group_id":      "123",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
			},
		},
	}, spans)
}

func TestNotStartingStreamingCreatesTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	agent := agents.New("test_agent").WithModelInstance(
		agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")},
		}),
	)

	result, err := agents.RunStreamed(t.Context(), agent, "first_test")
	require.NoError(t, err)

	t.Cleanup(func() {
		err = result.StreamEvents(func(agents.StreamEvent) error {
			return nil
		})
		require.NoError(t, err)
	})

	// Purposely don't await the stream
	for !result.IsComplete() {
		time.Sleep(10 * time.Millisecond)
	}

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
				},
			},
		},
	}, spans)
}

func TestStreamingSingleRunIsSingleTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	agent := agents.New("test_agent").WithModelInstance(
		agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")},
		}),
	)

	result, err := agents.RunStreamed(t.Context(), agent, "first_test")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

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
				},
			},
		},
	}, spans)
}

func TestMultipleStreamedRunsAreMultipleTraces(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("second_test")}},
	})
	agent := agents.New("test_agent_1").WithModelInstance(model)

	result, err := agents.RunStreamed(t.Context(), agent, "first_test")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	result, err = agents.RunStreamed(t.Context(), agent, "second_test")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
			},
		},
		{
			"workflow_name": "Agent workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
			},
		},
	}, spans)
}

func TestWrappedStreamingTraceIsSingleTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("second_test")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("third_test")}},
	})

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test_workflow"},
		func(ctx context.Context, _ tracing.Trace) error {
			agent := agents.New("test_agent_1").WithModelInstance(model)

			for _, input := range []string{"first_test", "second_test", "third_test"} {
				result, err := agents.RunStreamed(ctx, agent, input)
				if err != nil {
					return err
				}
				err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
				if err != nil {
					return err
				}
			}

			return nil
		})
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test_workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
			},
		},
	}, spans)
}

func TestWrappedMixedTraceIsSingleTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("second_test")}},
		{Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("third_test")}},
	})

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test_workflow"},
		func(ctx context.Context, _ tracing.Trace) error {
			agent := agents.New("test_agent_1").WithModelInstance(model)

			result, err := agents.RunStreamed(ctx, agent, "first_test")
			if err != nil {
				return err
			}
			err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
			if err != nil {
				return err
			}

			_, err = agents.Run(ctx, agent, "second_test")
			if err != nil {
				return err
			}

			result, err = agents.RunStreamed(ctx, agent, "third_test")
			if err != nil {
				return err
			}
			err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
			if err != nil {
				return err
			}

			return nil
		})
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test_workflow",
			"children": []m{
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
				{
					"type": "agent",
					"data": m{
						"name":        "test_agent_1",
						"handoffs":    []string{},
						"tools":       []string{},
						"output_type": "string",
					},
				},
			},
		},
	}, spans)
}

func TestParentDisabledTraceDisablesStreamingAgentTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test_workflow", Disabled: true},
		func(ctx context.Context, _ tracing.Trace) error {
			agent := agents.New("test_agent").WithModelInstance(
				agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
					Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")},
				}),
			)

			result, err := agents.RunStreamed(ctx, agent, "third_test")
			if err != nil {
				return err
			}
			err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
			if err != nil {
				return err
			}

			return nil
		})
	require.NoError(t, err)

	tracingtesting.RequireNoTraces(t)
}

func TestManualStreamingDisablingWorks(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	agent := agents.New("test_agent").WithModelInstance(
		agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("first_test")},
		}),
	)

	runner := agents.Runner{
		Config: agents.RunConfig{
			TracingDisabled: true,
		},
	}
	result, err := runner.RunStreamed(t.Context(), agent, "first_test")
	require.NoError(t, err)
	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
	require.NoError(t, err)

	tracingtesting.RequireNoTraces(t)
}
