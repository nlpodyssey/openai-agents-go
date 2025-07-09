package tracing_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/nlpodyssey/openai-agents-go/tracing/tracingtesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func tracingTestSetup(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()
}

func simpleTracing(t *testing.T) {
	ctx := t.Context()

	x := tracing.NewTrace(ctx, tracing.TraceParams{WorkflowName: "test"})
	require.NoError(t, x.Start(ctx, false))

	span1 := tracing.NewAgentSpan(ctx, tracing.AgentSpanParams{
		Name:   "agent_1",
		SpanID: "span_1",
		Parent: x,
	})
	require.NoError(t, span1.Start(ctx, false))
	require.NoError(t, span1.Finish(ctx, false))

	span2 := tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{
		Name:   "custom_1",
		SpanID: "span_2",
		Parent: x,
	})
	require.NoError(t, span2.Start(ctx, false))

	span3 := tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{
		Name:   "custom_2",
		SpanID: "span_3",
		Parent: span2,
	})
	require.NoError(t, span3.Start(ctx, false))
	require.NoError(t, span3.Finish(ctx, false))

	require.NoError(t, span2.Finish(ctx, false))

	require.NoError(t, x.Finish(ctx, false))
}

func TestSimpleTracing(t *testing.T) {
	tracingTestSetup(t)
	simpleTracing(t)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test",
			"children": []m{
				{
					"type": "agent",
					"id":   "span_1",
					"data": m{"name": "agent_1"},
				},
				{
					"type": "custom",
					"id":   "span_2",
					"data": m{"name": "custom_1"},
					"children": []m{
						{
							"type": "custom",
							"id":   "span_3",
							"data": m{"name": "custom_2"},
						},
					},
				},
			},
		},
	}, tracingtesting.FetchNormalizedSpans(t, true, false, false))
}

func ctxManagerSpans(t *testing.T) {
	ctx := t.Context()

	err := tracing.RunTrace(
		ctx, tracing.TraceParams{WorkflowName: "test", TraceID: "trace_123", GroupID: "456"},
		func(ctx context.Context, _ tracing.Trace) error {
			err := tracing.CustomSpan(
				ctx, tracing.CustomSpanParams{Name: "custom_1", SpanID: "span_1"},
				func(ctx context.Context, _ tracing.Span) error {
					return tracing.CustomSpan(
						ctx, tracing.CustomSpanParams{Name: "custom_2", SpanID: "span_1_inner"},
						func(context.Context, tracing.Span) error { return nil },
					)
				},
			)
			if err != nil {
				return err
			}

			return tracing.CustomSpan(
				ctx, tracing.CustomSpanParams{Name: "custom_2", SpanID: "span_2"},
				func(context.Context, tracing.Span) error { return nil },
			)
		},
	)
	require.NoError(t, err)
}

func TestCtxManagerSpans(t *testing.T) {
	tracingTestSetup(t)
	ctxManagerSpans(t)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test",
			"group_id":      "456",
			"children": []m{
				{
					"type": "custom",
					"id":   "span_1",
					"data": m{"name": "custom_1"},
					"children": []m{
						{
							"type": "custom",
							"id":   "span_1_inner",
							"data": m{"name": "custom_2"},
						},
					},
				},
				{"type": "custom", "id": "span_2", "data": m{"name": "custom_2"}},
			},
		},
	}, tracingtesting.FetchNormalizedSpans(t, true, false, false))
}

func runSubtask(ctx context.Context, spanID string) error {
	return tracing.GenerationSpan(
		ctx, tracing.GenerationSpanParams{SpanID: spanID},
		func(context.Context, tracing.Span) error {
			time.Sleep(100 * time.Microsecond)
			return nil
		},
	)
}

func simpleSubtasksTracing(t *testing.T) {
	ctx := t.Context()

	err := tracing.RunTrace(
		ctx, tracing.TraceParams{WorkflowName: "test", TraceID: "trace_123", GroupID: "group_456"},
		func(ctx context.Context, _ tracing.Trace) error {
			if err := runSubtask(ctx, "span_1"); err != nil {
				return err
			}
			return runSubtask(ctx, "span_2")
		},
	)
	require.NoError(t, err)
}

func TestSimpleSubtasksTracing(t *testing.T) {
	tracingTestSetup(t)
	simpleSubtasksTracing(t)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test",
			"group_id":      "group_456",
			"children": []m{
				{"type": "generation", "id": "span_1"},
				{"type": "generation", "id": "span_2"},
			},
		},
	}, tracingtesting.FetchNormalizedSpans(t, true, false, false))
}

func runTasksParallel(ctx context.Context, spanIDs []string) error {
	errs := make([]error, len(spanIDs))

	var wg sync.WaitGroup
	wg.Add(len(spanIDs))

	for i, spanID := range spanIDs {
		go func() {
			defer wg.Done()
			errs[i] = runSubtask(ctx, spanID)
		}()
	}

	wg.Wait()
	return errors.Join(errs...)
}

func runTasksAsChildren(ctx context.Context, firstSpanID, secondSpanID string) error {
	return tracing.GenerationSpan(
		ctx, tracing.GenerationSpanParams{SpanID: firstSpanID},
		func(ctx context.Context, _ tracing.Span) error {
			return runSubtask(ctx, secondSpanID)
		},
	)
}

func complexAsyncTracing(t *testing.T) {
	ctx := t.Context()

	err := tracing.RunTrace(
		ctx, tracing.TraceParams{WorkflowName: "test", TraceID: "trace_123", GroupID: "456"},
		func(ctx context.Context, _ tracing.Trace) error {
			var errs [2]error
			var wg sync.WaitGroup

			wg.Add(2)
			go func() {
				defer wg.Done()
				errs[0] = runTasksParallel(ctx, []string{"span_1", "span_2"})
			}()
			go func() {
				defer wg.Done()
				errs[1] = runTasksParallel(ctx, []string{"span_3", "span_4"})
			}()

			wg.Wait()
			if err := errors.Join(errs[:]...); err != nil {
				return err
			}

			wg.Add(2)
			go func() {
				defer wg.Done()
				errs[0] = runTasksAsChildren(ctx, "span_5", "span_6")
			}()
			go func() {
				defer wg.Done()
				errs[1] = runTasksAsChildren(ctx, "span_7", "span_8")
			}()

			wg.Wait()
			if err := errors.Join(errs[:]...); err != nil {
				return err
			}

			return nil
		},
	)

	require.NoError(t, err)
}

func TestComplexAsyncTracing(t *testing.T) {
	tracingTestSetup(t)

	for range 300 {
		tracingtesting.SpanProcessorTesting().Clear()
		complexAsyncTracing(t)

		type m = map[string]any
		require.Equal(t, []m{
			{
				"workflow_name": "test",
				"group_id":      "456",
				"children": []m{
					m{"type": "generation", "id": "span_1"},
					m{"type": "generation", "id": "span_2"},
					m{"type": "generation", "id": "span_3"},
					m{"type": "generation", "id": "span_4"},
					m{
						"type":     "generation",
						"id":       "span_5",
						"children": []m{{"type": "generation", "id": "span_6"}},
					},
					m{
						"type":     "generation",
						"id":       "span_7",
						"children": []m{{"type": "generation", "id": "span_8"}},
					},
				},
			},
		}, tracingtesting.FetchNormalizedSpans(t, true, false, true))
	}
}

func spansWithSetters(t *testing.T) {
	ctx := t.Context()

	err := tracing.RunTrace(
		ctx, tracing.TraceParams{WorkflowName: "test", TraceID: "trace_123", GroupID: "456"},
		func(ctx context.Context, _ tracing.Trace) error {
			return tracing.AgentSpan(
				ctx, tracing.AgentSpanParams{Name: "agent_1"},
				func(ctx context.Context, agentSpan tracing.Span) error {
					agentSpan.SpanData().(*tracing.AgentSpanData).Name = "agent_2"

					err := tracing.FunctionSpan(
						ctx, tracing.FunctionSpanParams{Name: "function_1"},
						func(_ context.Context, functionSpan tracing.Span) error {
							data := functionSpan.SpanData().(*tracing.FunctionSpanData)
							data.Input = "i"
							data.Output = "o"
							return nil
						},
					)
					if err != nil {
						return err
					}

					err = tracing.GenerationSpan(
						ctx, tracing.GenerationSpanParams{},
						func(_ context.Context, generationSpan tracing.Span) error {
							generationSpan.SpanData().(*tracing.GenerationSpanData).Input = []map[string]any{
								{"foo": "bar"},
							}
							return nil
						},
					)
					if err != nil {
						return err
					}

					return tracing.HandoffSpan(
						ctx, tracing.HandoffSpanParams{FromAgent: "agent_1", ToAgent: "agent_2"},
						func(context.Context, tracing.Span) error { return nil },
					)
				},
			)
		},
	)
	require.NoError(t, err)
}

func TestSpansWithSetters(t *testing.T) {
	tracingTestSetup(t)
	spansWithSetters(t)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test",
			"group_id":      "456",
			"children": []m{
				{
					"type": "agent",
					"data": m{"name": "agent_2"},
					"children": []m{
						{
							"type": "function",
							"data": m{"name": "function_1", "input": "i", "output": "o"},
						},
						{
							"type": "generation",
							"data": m{"input": []m{{"foo": "bar"}}},
						},
						{
							"type": "handoff",
							"data": m{"from_agent": "agent_1", "to_agent": "agent_2"},
						},
					},
				},
			},
		},
	}, tracingtesting.FetchNormalizedSpans(t, false, false, false))
}

func disabledTracing(t *testing.T) {
	ctx := t.Context()

	err := tracing.RunTrace(
		ctx, tracing.TraceParams{WorkflowName: "test", TraceID: "123", GroupID: "456", Disabled: true},
		func(ctx context.Context, _ tracing.Trace) error {
			return tracing.AgentSpan(
				ctx, tracing.AgentSpanParams{Name: "agent_1"},
				func(ctx context.Context, _ tracing.Span) error {
					return tracing.FunctionSpan(
						ctx, tracing.FunctionSpanParams{Name: "function_1"},
						func(context.Context, tracing.Span) error { return nil },
					)
				},
			)
		},
	)

	require.NoError(t, err)
}

func TestDisabledTracing(t *testing.T) {
	tracingTestSetup(t)
	disabledTracing(t)
	tracingtesting.RequireNoTraces(t)
}

func enabledTraceDisabledSpan(t *testing.T) {
	ctx := t.Context()

	err := tracing.RunTrace(
		ctx, tracing.TraceParams{WorkflowName: "test", TraceID: "trace_123"},
		func(ctx context.Context, _ tracing.Trace) error {
			return tracing.AgentSpan(
				ctx, tracing.AgentSpanParams{Name: "agent_1"},
				func(ctx context.Context, _ tracing.Span) error {
					return tracing.FunctionSpan(
						ctx, tracing.FunctionSpanParams{Name: "function_1", Disabled: true},
						func(ctx context.Context, spa_ tracing.Span) error {
							return tracing.GenerationSpan(
								ctx, tracing.GenerationSpanParams{},
								func(context.Context, tracing.Span) error { return nil },
							)
						},
					)
				},
			)
		},
	)

	require.NoError(t, err)
}

func TestEnabledTraceDisabledSpan(t *testing.T) {
	tracingTestSetup(t)
	enabledTraceDisabledSpan(t)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test",
			"children": []m{
				{
					"type": "agent",
					"data": m{"name": "agent_1"},
				},
			},
		},
	}, tracingtesting.FetchNormalizedSpans(t, false, false, false))
}

func TestStartAndEndCalledManual(t *testing.T) {
	tracingTestSetup(t)
	simpleTracing(t)

	events := tracingtesting.FetchEvents()
	assert.Equal(t, []tracingtesting.SpanProcessorEvent{
		tracingtesting.TraceStart,
		tracingtesting.SpanStart, // span_1
		tracingtesting.SpanEnd,   // span_1
		tracingtesting.SpanStart, // span_2
		tracingtesting.SpanStart, // span_3
		tracingtesting.SpanEnd,   // span_3
		tracingtesting.SpanEnd,   // span_2
		tracingtesting.TraceEnd,
	}, events)
}

func TestStartAndEndCalledCtxManager(t *testing.T) {
	tracingTestSetup(t)

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test", TraceID: "123", GroupID: "456"},
		func(ctx context.Context, _ tracing.Trace) error {
			err := tracing.CustomSpan(
				ctx, tracing.CustomSpanParams{Name: "custom_1", SpanID: "span_1"},
				func(ctx context.Context, _ tracing.Span) error {
					return tracing.CustomSpan(
						ctx, tracing.CustomSpanParams{Name: "custom_2", SpanID: "span_1_inner"},
						func(ctx context.Context, _ tracing.Span) error { return nil },
					)
				},
			)
			if err != nil {
				return err
			}

			return tracing.CustomSpan(
				ctx, tracing.CustomSpanParams{Name: "custom_2", SpanID: "span_2"},
				func(ctx context.Context, span tracing.Span) error { return nil },
			)
		},
	)
	require.NoError(t, err)

	events := tracingtesting.FetchEvents()
	assert.Equal(t, []tracingtesting.SpanProcessorEvent{
		tracingtesting.TraceStart,
		tracingtesting.SpanStart, // span_1
		tracingtesting.SpanStart, // span_1_inner
		tracingtesting.SpanEnd,   // span_1_inner
		tracingtesting.SpanEnd,   // span_1
		tracingtesting.SpanStart, // span_2
		tracingtesting.SpanEnd,   // span_2
		tracingtesting.TraceEnd,
	}, events)
}

func TestStartAndEndCalledSubtasksCtxManager(t *testing.T) {
	tracingTestSetup(t)
	simpleSubtasksTracing(t)

	events := tracingtesting.FetchEvents()
	assert.Equal(t, []tracingtesting.SpanProcessorEvent{
		tracingtesting.TraceStart,
		tracingtesting.SpanStart, // span_1
		tracingtesting.SpanEnd,   // span_1
		tracingtesting.SpanStart, // span_2
		tracingtesting.SpanEnd,   // span_2
		tracingtesting.TraceEnd,
	}, events)
}

func TestNoopSpanDoesntRecord(t *testing.T) {
	tracingTestSetup(t)

	ctx := t.Context()

	trace := tracing.NewTrace(ctx, tracing.TraceParams{WorkflowName: "test", Disabled: true})
	var span tracing.Span

	err := trace.Run(ctx, func(ctx context.Context, _ tracing.Trace) error {
		span = tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{Name: "span_1"})
		return span.Run(ctx, func(_ context.Context, span tracing.Span) error {
			span.SetError(tracing.SpanError{
				Message: "test",
				Data:    nil,
			})
			return nil
		},
		)
	},
	)
	require.NoError(t, err)

	tracingtesting.RequireNoTraces(t)
	assert.Nil(t, trace.Export())
	assert.Nil(t, span.Export())
	assert.Zero(t, span.StartedAt())
	assert.Zero(t, span.EndedAt())
	assert.Nil(t, span.Error())
}

func TestMultipleSpanStartFinishDoesntError(t *testing.T) {
	tracingTestSetup(t)

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test", TraceID: "123", GroupID: "456"},
		func(ctx context.Context, _ tracing.Trace) error {
			span := tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{Name: "span_1"})
			err := span.Run(ctx, func(ctx context.Context, s tracing.Span) error {
				return s.Start(ctx, false)
			})
			if err != nil {
				return err
			}
			return span.Finish(ctx, false)
		})
	require.NoError(t, err)
}

func TestNoopParentIsNoopChild(t *testing.T) {
	tracingTestSetup(t)
	ctx := t.Context()

	tr := tracing.NewTrace(ctx, tracing.TraceParams{WorkflowName: "test", Disabled: true})

	span := tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{Name: "span_1", Parent: tr})
	require.NoError(t, span.Start(ctx, false))
	require.NoError(t, span.Finish(ctx, false))

	require.Nil(t, span.Export())

	span2 := tracing.NewCustomSpan(ctx, tracing.CustomSpanParams{Name: "span_2", Parent: span})
	require.NoError(t, span2.Start(ctx, false))
	require.NoError(t, span2.Finish(ctx, false))

	require.Nil(t, span2.Export())
}
