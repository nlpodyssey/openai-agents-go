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

package tracing

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBackendSpanExporter_APIKey(t *testing.T) {
	t.Run("SetAPIKey", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "")

		// If the API key is not set, it should stay empty string
		processor := NewBackendSpanExporter(BackendSpanExporterParams{})
		assert.Equal(t, "", processor.APIKey())

		// If we set it afterward, it should be the new value
		processor.SetAPIKey("test_api_key")
		assert.Equal(t, "test_api_key", processor.APIKey())
	})

	t.Run("from env", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "")

		// If the API key is not set at creation time but set before access
		// time, it should be the new value.
		processor := NewBackendSpanExporter(BackendSpanExporterParams{})
		assert.Equal(t, "", processor.APIKey())

		// If we set it afterward, it should be the new value
		t.Setenv("OPENAI_API_KEY", "foo_bar_123")
		assert.Equal(t, "foo_bar_123", processor.APIKey())
	})
}

// getSpan creates a minimal agent span for testing processors.
func getSpan(processor Processor) *SpanImpl {
	return NewSpanImpl(
		"test_trace_id",
		"test_span_id",
		"",
		processor,
		&AgentSpanData{Name: "test_agent"},
	)
}

// getTrace creates a minimal trace.
func getTrace(processor Processor) *TraceImpl {
	return NewTraceImpl(
		"test_trace",
		"test_trace_id",
		"test_session_id",
		nil,
		processor,
	)
}

func queueSize(processor *BatchTraceProcessor) int {
	processor.queueMu.RLock()
	defer processor.queueMu.RUnlock()
	return processor.queueSize
}

type mockedExporter struct {
	callItems [][]any
}

func (e *mockedExporter) Export(_ context.Context, items []any) error {
	e.callItems = append(e.callItems, items)
	return nil
}

func TestBatchTraceProcessor_OnTraceStart(t *testing.T) {
	ctx := t.Context()

	processor := NewBatchTraceProcessor(BatchTraceProcessorParams{
		Exporter:      &mockedExporter{},
		ScheduleDelay: param.NewOpt(100 * time.Millisecond),
	})
	t.Cleanup(func() { require.NoError(t, processor.Shutdown(ctx)) })

	testTrace := getTrace(processor)

	err := processor.OnTraceStart(ctx, testTrace)
	require.NoError(t, err)

	assert.Equal(t, 1, queueSize(processor))
}

func TestBatchTraceProcessor_OnSpanEnd(t *testing.T) {
	ctx := t.Context()

	processor := NewBatchTraceProcessor(BatchTraceProcessorParams{
		Exporter:      &mockedExporter{},
		ScheduleDelay: param.NewOpt(100 * time.Millisecond),
	})
	t.Cleanup(func() { require.NoError(t, processor.Shutdown(ctx)) })

	testSpan := getSpan(processor)

	err := processor.OnSpanEnd(ctx, testSpan)
	require.NoError(t, err)

	assert.Equal(t, 1, queueSize(processor))
}

func TestBatchTraceProcessorQueueFull(t *testing.T) {
	ctx := t.Context()

	processor := NewBatchTraceProcessor(BatchTraceProcessorParams{
		Exporter:           &mockedExporter{},
		MaxQueueSize:       param.NewOpt(2),
		ScheduleDelay:      param.NewOpt(5 * time.Second),
		ExportTriggerRatio: param.NewOpt(2.0),
	})
	t.Cleanup(func() { require.NoError(t, processor.Shutdown(ctx)) })

	// Fill the queue
	require.NoError(t, processor.OnTraceStart(ctx, getTrace(processor)))
	require.NoError(t, processor.OnTraceStart(ctx, getTrace(processor)))
	assert.Equal(t, 2, queueSize(processor))

	// Next item should not be queued
	require.NoError(t, processor.OnTraceStart(ctx, getTrace(processor)))
	assert.Equal(t, 2, queueSize(processor))

	require.NoError(t, processor.OnSpanEnd(ctx, getSpan(processor)))
	assert.Equal(t, 2, queueSize(processor))
}

func TestBatchProcessorDoesntEnqueueOnTraceEndOrSpanStart(t *testing.T) {
	ctx := t.Context()

	processor := NewBatchTraceProcessor(BatchTraceProcessorParams{Exporter: &mockedExporter{}})
	t.Cleanup(func() { require.NoError(t, processor.Shutdown(ctx)) })

	require.NoError(t, processor.OnTraceStart(ctx, getTrace(processor)))
	assert.Equal(t, 1, queueSize(processor), "Trace should be queued")

	require.NoError(t, processor.OnSpanStart(ctx, getSpan(processor)))
	assert.Equal(t, 1, queueSize(processor), "Span should not be queued")

	require.NoError(t, processor.OnSpanEnd(ctx, getSpan(processor)))
	assert.Equal(t, 2, queueSize(processor), "Span should be queued")

	require.NoError(t, processor.OnTraceEnd(ctx, getTrace(processor)))
	assert.Equal(t, 2, queueSize(processor), "Nothing new should be queued")
}

func TestBatchTraceProcessorForceFlush(t *testing.T) {
	ctx := t.Context()

	exporter := &mockedExporter{}
	processor := NewBatchTraceProcessor(BatchTraceProcessorParams{
		Exporter:      exporter,
		MaxBatchSize:  param.NewOpt(2),
		ScheduleDelay: param.NewOpt(5 * time.Second),
	})
	t.Cleanup(func() { require.NoError(t, processor.Shutdown(ctx)) })

	require.NoError(t, processor.OnTraceStart(ctx, getTrace(processor)))
	require.NoError(t, processor.OnSpanEnd(ctx, getSpan(processor)))
	require.NoError(t, processor.OnSpanEnd(ctx, getSpan(processor)))

	require.NoError(t, processor.ForceFlush(ctx))

	// Ensure exporter.Export was called with all items
	// Because MaxBatchSize=2, it may have been called multiple times
	totalExported := 0
	for _, batch := range exporter.callItems {
		totalExported += len(batch)
	}

	// We pushed 3 items; ensure they all got exported
	assert.Equal(t, 3, totalExported)
}

func TestBatchTraceProcessorShutdownFlushes(t *testing.T) {
	ctx := t.Context()

	exporter := &mockedExporter{}
	processor := NewBatchTraceProcessor(BatchTraceProcessorParams{
		Exporter:      exporter,
		ScheduleDelay: param.NewOpt(5 * time.Second),
	})

	require.NoError(t, processor.OnTraceStart(ctx, getTrace(processor)))
	require.NoError(t, processor.OnSpanEnd(ctx, getSpan(processor)))

	queueSizeBefore := queueSize(processor)
	assert.Equal(t, 2, queueSizeBefore)

	require.NoError(t, processor.Shutdown(ctx))

	// Ensure everything was exported after shutdown
	totalExported := 0
	for _, batch := range exporter.callItems {
		totalExported += len(batch)
	}

	assert.Equal(t, 2, totalExported, "All items in the queue should be exported upon shutdown")
}

func TestBatchTraceProcessorScheduledExport(t *testing.T) {
	// Tests that items are automatically exported when the ScheduleDelay expires.

	ctx := t.Context()

	exporter := &mockedExporter{}
	processor := NewBatchTraceProcessor(BatchTraceProcessorParams{
		Exporter:      exporter,
		ScheduleDelay: param.NewOpt(100 * time.Millisecond),
	})

	require.NoError(t, processor.OnSpanEnd(ctx, getSpan(processor))) // queue size = 1

	//  Now advance time beyond the next export time and let the background thread run a bit
	time.Sleep(300 * time.Millisecond)

	// Check that exporter.Export was eventually called
	// Because the background thread runs, we might need a small sleep
	require.NoError(t, processor.Shutdown(ctx))

	totalExported := 0
	for _, batch := range exporter.callItems {
		totalExported += len(batch)
	}

	assert.Equal(t, 1, totalExported, "Item should be exported after scheduled delay")
}

type noOpProcessor struct{}

func (noOpProcessor) OnTraceStart(context.Context, Trace) error { return nil }
func (noOpProcessor) OnTraceEnd(context.Context, Trace) error   { return nil }
func (noOpProcessor) OnSpanStart(context.Context, Span) error   { return nil }
func (noOpProcessor) OnSpanEnd(context.Context, Span) error     { return nil }
func (noOpProcessor) Shutdown(context.Context) error            { return nil }
func (noOpProcessor) ForceFlush(context.Context) error          { return nil }

type testingTransport struct {
	response *http.Response
	err      error
	requests []*http.Request
	closed   bool
}

func (r *testingTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	r.requests = append(r.requests, request)
	switch {
	case r.err != nil:
		return nil, r.err
	case r.response != nil:
		return r.response, nil
	default:
		return &http.Response{Status: "OK", StatusCode: 200}, nil
	}
}

func (r *testingTransport) CloseIdleConnections() {
	r.closed = true
}

func TestBackendSpanExporterNoItems(t *testing.T) {
	rt := &testingTransport{}
	exporter := NewBackendSpanExporter(BackendSpanExporterParams{
		APIKey:     "test_key",
		HTTPClient: &http.Client{Transport: rt},
	})
	t.Cleanup(func() { exporter.Close() })

	require.NoError(t, exporter.Export(t.Context(), nil))
	require.NoError(t, exporter.Export(t.Context(), []any{}))

	// No calls should be made if there are no items
	assert.Empty(t, rt.requests)
}

func TestBackendSpanExporterNoAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")

	rt := &testingTransport{}
	exporter := NewBackendSpanExporter(BackendSpanExporterParams{
		APIKey:     "",
		HTTPClient: &http.Client{Transport: rt},
	})
	t.Cleanup(func() { exporter.Close() })

	require.NoError(t, exporter.Export(t.Context(), []any{getSpan(noOpProcessor{})}))

	// Should log an error and return without calling post
	assert.Empty(t, rt.requests)
}

func TestBackendSpanExporter2xxSuccess(t *testing.T) {
	rt := &testingTransport{
		response: &http.Response{StatusCode: 200},
	}
	exporter := NewBackendSpanExporter(BackendSpanExporterParams{
		APIKey:     "test_key",
		HTTPClient: &http.Client{Transport: rt},
	})
	t.Cleanup(func() { exporter.Close() })

	require.NoError(t, exporter.Export(t.Context(), []any{
		getSpan(noOpProcessor{}), getTrace(noOpProcessor{}),
	}))

	// Should have called post exactly once
	assert.Len(t, rt.requests, 1)
}

func TestBackendSpanExporter4xxClientError(t *testing.T) {
	rt := &testingTransport{
		response: &http.Response{StatusCode: 400, Status: "Bad Request"},
	}
	exporter := NewBackendSpanExporter(BackendSpanExporterParams{
		APIKey:     "test_key",
		HTTPClient: &http.Client{Transport: rt},
	})
	t.Cleanup(func() { exporter.Close() })

	require.NoError(t, exporter.Export(t.Context(), []any{getSpan(noOpProcessor{})}))

	// 4xx should not be retried
	assert.Len(t, rt.requests, 1)
}

func TestBackendSpanExporter5xxRetry(t *testing.T) {
	rt := &testingTransport{
		response: &http.Response{StatusCode: 500},
	}
	exporter := NewBackendSpanExporter(BackendSpanExporterParams{
		APIKey:     "test_key",
		MaxRetries: param.NewOpt(3),
		BaseDelay:  param.NewOpt(50 * time.Millisecond),
		MaxDelay:   param.NewOpt(100 * time.Millisecond),
		HTTPClient: &http.Client{Transport: rt},
	})
	t.Cleanup(func() { exporter.Close() })

	require.NoError(t, exporter.Export(t.Context(), []any{getSpan(noOpProcessor{})}))

	// Should retry up to MaxRetries times
	assert.Len(t, rt.requests, 3)
}

func TestBackendSpanExporterRequestError(t *testing.T) {
	rt := &testingTransport{
		err: errors.New("error"),
	}
	exporter := NewBackendSpanExporter(BackendSpanExporterParams{
		APIKey:     "test_key",
		MaxRetries: param.NewOpt(2),
		BaseDelay:  param.NewOpt(50 * time.Millisecond),
		MaxDelay:   param.NewOpt(100 * time.Millisecond),
		HTTPClient: &http.Client{Transport: rt},
	})
	t.Cleanup(func() { exporter.Close() })

	require.NoError(t, exporter.Export(t.Context(), []any{getSpan(noOpProcessor{})}))

	// Should retry up to MaxRetries times
	assert.Len(t, rt.requests, 2)
}

func TestBackendSpanExporterClose(t *testing.T) {
	rt := &testingTransport{}
	exporter := NewBackendSpanExporter(BackendSpanExporterParams{
		APIKey:     "test_key",
		HTTPClient: &http.Client{Transport: rt},
	})
	exporter.Close()

	// Ensure underlying http client is closed
	assert.True(t, rt.closed)
}
