package agents_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/nlpodyssey/openai-agents-go/tracing/tracingtesting"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newDummyClient(t *testing.T, body string) agents.OpenaiClient {
	t.Helper()

	return agents.OpenaiClient{
		Client: openai.NewClient(
			option.WithMiddleware(func(req *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
				return &http.Response{
					StatusCode:    http.StatusOK,
					Body:          io.NopCloser(strings.NewReader(body)),
					ContentLength: int64(len(body)),
					Header:        http.Header{"Content-Type": []string{"application/json"}},
				}, nil
			}),
		),
	}
}

func newDummyClientNonStreaming(t *testing.T) agents.OpenaiClient {
	t.Helper()
	return newDummyClient(t, `{ "id": "dummy-id" }`)
}

func newDummyClientStreaming(t *testing.T) agents.OpenaiClient {
	t.Helper()
	return newDummyClient(t, `event: response.completed
data: {"type":"response.completed","response":{"id":"dummy-id-123"}}

`)
}

func TestGetResponseCreatesTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test"},
		func(ctx context.Context, _ tracing.Trace) error {
			model := agents.NewOpenAIResponsesModel("test-model", newDummyClientNonStreaming(t))

			_, err := model.GetResponse(ctx, agents.ModelResponseParams{
				SystemInstructions: param.NewOpt("instr"),
				Input:              agents.InputString("input"),
				Tracing:            agents.ModelTracingEnabled,
			})
			return err
		})
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test",
			"children":      []m{{"type": "response", "data": m{"response_id": "dummy-id"}}},
		},
	}, spans)
}

func TestNonDataTracingDoesNotSetResponseID(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test"},
		func(ctx context.Context, _ tracing.Trace) error {
			model := agents.NewOpenAIResponsesModel("test-model", newDummyClientNonStreaming(t))

			_, err := model.GetResponse(ctx, agents.ModelResponseParams{
				SystemInstructions: param.NewOpt("instr"),
				Input:              agents.InputString("input"),
				Tracing:            agents.ModelTracingEnabledWithoutData,
			})
			return err
		})
	require.NoError(t, err)

	normalizedSpans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{"workflow_name": "test", "children": []m{{"type": "response"}}},
	}, normalizedSpans)

	orderedSpans := tracingtesting.FetchOrderedSpans(false)
	assert.Nil(t, orderedSpans[0].SpanData().(*tracing.ResponseSpanData).Response)
}

func TestDisableTracingDoesNotCreateSpan(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test"},
		func(ctx context.Context, _ tracing.Trace) error {
			model := agents.NewOpenAIResponsesModel("test-model", newDummyClientNonStreaming(t))

			_, err := model.GetResponse(ctx, agents.ModelResponseParams{
				SystemInstructions: param.NewOpt("instr"),
				Input:              agents.InputString("input"),
				Tracing:            agents.ModelTracingDisabled,
			})
			return err
		})
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{{"workflow_name": "test"}}, spans)

	tracingtesting.RequireNoSpans(t)
}

func TestStreamResponseCreatesTrace(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test"},
		func(ctx context.Context, _ tracing.Trace) error {
			model := agents.NewOpenAIResponsesModel("test-model", newDummyClientStreaming(t))

			return model.StreamResponse(
				ctx,
				agents.ModelResponseParams{
					SystemInstructions: param.NewOpt("instr"),
					Input:              agents.InputString("input"),
					Tracing:            agents.ModelTracingEnabled,
				},
				func(context.Context, agents.TResponseStreamEvent) error {
					return nil
				},
			)
		})
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{
			"workflow_name": "test",
			"children":      []m{{"type": "response", "data": m{"response_id": "dummy-id-123"}}},
		},
	}, spans)
}

func TestStreamNonDataTracingDoesNotSetResponseID(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test"},
		func(ctx context.Context, _ tracing.Trace) error {
			model := agents.NewOpenAIResponsesModel("test-model", newDummyClientStreaming(t))

			return model.StreamResponse(
				ctx,
				agents.ModelResponseParams{
					SystemInstructions: param.NewOpt("instr"),
					Input:              agents.InputString("input"),
					Tracing:            agents.ModelTracingEnabledWithoutData,
				},
				func(context.Context, agents.TResponseStreamEvent) error {
					return nil
				},
			)
		})
	require.NoError(t, err)

	normalizedSpans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{
		{"workflow_name": "test", "children": []m{{"type": "response"}}},
	}, normalizedSpans)

	orderedSpans := tracingtesting.FetchOrderedSpans(false)
	assert.Nil(t, orderedSpans[0].SpanData().(*tracing.ResponseSpanData).Response)
}

func TestStreamDisabledTracingDoesNotCreateSpan(t *testing.T) {
	tracingtesting.Setup(t)
	agents.ClearOpenaiSettings()

	err := tracing.RunTrace(
		t.Context(), tracing.TraceParams{WorkflowName: "test"},
		func(ctx context.Context, _ tracing.Trace) error {
			model := agents.NewOpenAIResponsesModel("test-model", newDummyClientStreaming(t))

			return model.StreamResponse(
				ctx,
				agents.ModelResponseParams{
					SystemInstructions: param.NewOpt("instr"),
					Input:              agents.InputString("input"),
					Tracing:            agents.ModelTracingDisabled,
				}, func(context.Context, agents.TResponseStreamEvent) error {
					return nil
				},
			)
		})
	require.NoError(t, err)

	spans := tracingtesting.FetchNormalizedSpans(t, false, false, false)

	type m = map[string]any
	assert.Equal(t, []m{{"workflow_name": "test"}}, spans)

	tracingtesting.RequireNoSpans(t)
}
