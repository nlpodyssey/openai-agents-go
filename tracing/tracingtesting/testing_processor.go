package tracingtesting

import (
	"cmp"
	"context"
	"maps"
	"reflect"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nlpodyssey/openai-agents-go/tracing"
)

type SpanProcessorEvent string

const (
	TraceStart SpanProcessorEvent = "trace_start"
	TraceEnd   SpanProcessorEvent = "trace_end"
	SpanStart  SpanProcessorEvent = "span_start"
	SpanEnd    SpanProcessorEvent = "span_end"
)

// SpanProcessorForTests is a simple processor that stores finished spans in memory.
// This is concurrency-safe and suitable for tests or basic usage.
type SpanProcessorForTests struct {
	mu     sync.RWMutex
	spans  []tracing.Span
	traces []tracing.Trace
	events []SpanProcessorEvent
}

func NewSpanProcessorForTests() *SpanProcessorForTests {
	return &SpanProcessorForTests{}
}

func (p *SpanProcessorForTests) OnTraceStart(_ context.Context, trace tracing.Trace) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.traces = append(p.traces, trace)
	p.events = append(p.events, TraceStart)

	return nil
}

func (p *SpanProcessorForTests) OnTraceEnd(context.Context, tracing.Trace) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// We don't append the trace here, we want to do that in OnTraceStart
	p.events = append(p.events, TraceEnd)

	return nil
}

func (p *SpanProcessorForTests) OnSpanStart(context.Context, tracing.Span) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Purposely not appending the span here, we want to do that in OnSpanEnd
	p.events = append(p.events, SpanStart)

	return nil
}

func (p *SpanProcessorForTests) OnSpanEnd(_ context.Context, span tracing.Span) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.events = append(p.events, SpanEnd)
	p.spans = append(p.spans, span)

	return nil
}

func (p *SpanProcessorForTests) Shutdown(context.Context) error {
	return nil
}

func (p *SpanProcessorForTests) ForceFlush(context.Context) error {
	return nil
}

func (p *SpanProcessorForTests) GetOrderedSpans(includingEmpty, sortSpansByID bool) []tracing.Span {
	p.mu.RLock()
	defer p.mu.RUnlock()

	spans := slices.Clone(p.spans)
	if !includingEmpty {
		spans = slices.DeleteFunc(spans, func(span tracing.Span) bool {
			return len(span.Export()) == 0
		})
	}

	if sortSpansByID {
		sort.Slice(spans, func(i, j int) bool {
			return cmp.Less(spans[i].SpanID(), spans[j].SpanID())
		})
	} else {
		sort.Slice(spans, func(i, j int) bool {
			return spans[i].StartedAt().Before(spans[j].StartedAt())
		})
	}

	return spans
}

func (p *SpanProcessorForTests) GetTraces(includingEmpty bool) []tracing.Trace {
	p.mu.RLock()
	defer p.mu.RUnlock()

	traces := slices.Clone(p.traces)
	if !includingEmpty {
		traces = slices.DeleteFunc(traces, func(trace tracing.Trace) bool {
			return len(trace.Export()) == 0
		})
	}

	return traces
}

func (p *SpanProcessorForTests) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.spans = nil
	p.traces = nil
	p.events = nil
}

var spanProcessorTesting = NewSpanProcessorForTests()

func SpanProcessorTesting() *SpanProcessorForTests {
	return spanProcessorTesting
}

func FetchOrderedSpans(sortSpansByID bool) []tracing.Span {
	return SpanProcessorTesting().GetOrderedSpans(false, sortSpansByID)
}

func FetchTraces() []tracing.Trace {
	return SpanProcessorTesting().GetTraces(false)
}

func FetchEvents() []SpanProcessorEvent {
	p := SpanProcessorTesting()

	p.mu.RLock()
	defer p.mu.RUnlock()

	return slices.Clone(p.events)
}

func RequireNoSpans(t *testing.T) {
	t.Helper()

	spans := FetchOrderedSpans(false)
	if len(spans) > 0 {
		t.Fatalf("expected 0 spans, got %d", len(spans))
	}
}

func RequireNoTraces(t *testing.T) {
	t.Helper()

	traces := FetchTraces()
	if len(traces) > 0 {
		t.Fatalf("expected 0 traces, got %d", len(traces))
	}

	RequireNoSpans(t)
}

func FetchNormalizedSpans(t *testing.T, keepSpanID, keepTraceID, sortSpansByID bool) []map[string]any {
	t.Helper()

	nodes := make(map[[2]string]map[string]any)
	var traces []map[string]any

	for _, traceObj := range FetchTraces() {
		trace := traceObj.Export()

		if len(trace) == 0 {
			t.Fatal("trace must not be empty")
		}

		if v := trace["object"]; v != "trace" {
			t.Fatalf(`expected "object": "trace" in trace %+v`, trace)
		}
		delete(trace, "object")

		if v, ok := trace["id"].(string); !ok || !strings.HasPrefix(v, "trace_") {
			t.Fatalf(`expected "id": "trace_..." in trace %+v`, trace)
		}
		if !keepTraceID {
			delete(trace, "id")
		}

		deleteNilFromMap(trace)
		nodes[[2]string{traceObj.TraceID(), ""}] = trace
		traces = append(traces, trace)
	}

	if len(traces) == 0 {
		t.Fatal("expected traces, got none (use RequireNoTraces() to check for empty traces)")
	}

	for _, spanObj := range FetchOrderedSpans(sortSpansByID) {
		span := spanObj.Export()

		if len(span) == 0 {
			t.Fatal("span must not be empty")
		}

		if v := span["object"]; v != "trace.span" {
			t.Fatalf(`expected "object": "trace.span" in span %+v`, span)
		}
		delete(span, "object")

		if v, ok := span["id"].(string); !ok || !strings.HasPrefix(v, "span_") {
			t.Fatalf(`expected "id": "span_..." in span %+v`, span)
		}
		if !keepSpanID {
			delete(span, "id")
		}

		if v, ok := span["started_at"].(string); !ok || !canParseRFC3339NanoTime(v) {
			t.Fatalf(`expected "started_at" RFC3339Nano time value in span %+v`, span)
		}
		delete(span, "started_at")

		if v, ok := span["ended_at"].(string); !ok || !canParseRFC3339NanoTime(v) {
			t.Fatalf(`expected "ended_at" RFC3339Nano time value in span %+v`, span)
		}
		delete(span, "ended_at")

		if _, ok := span["parent_id"]; !ok {
			t.Fatalf(`expected "parent_id" in span %+v`, span)
		}
		parentID, _ := span["parent_id"].(string)
		delete(span, "parent_id")

		if _, ok := span["type"]; ok {
			t.Fatalf(`unexpected "type" in span %+v`, span)
		}

		spanData, ok := span["span_data"].(map[string]any)
		if !ok {
			t.Fatalf(`expected map[string]any "span_data" in span %+v`, span)
		}
		delete(span, "span_data")

		span["type"] = spanData["type"]
		delete(spanData, "type")

		deleteNilFromMap(span)
		deleteNilFromMap(spanData)

		if len(spanData) > 0 {
			span["data"] = spanData
		}

		nodes[[2]string{spanObj.TraceID(), spanObj.SpanID()}] = span

		traceID, ok := span["trace_id"].(string)
		if !ok {
			t.Fatalf(`expected string "trace_id" in span %+v`, span)
		}
		delete(span, "trace_id")
		node, ok := nodes[[2]string{traceID, parentID}]
		if !ok {
			t.Fatalf("node for [%q, %q] not found", traceID, parentID)
		}
		children, _ := node["children"].([]map[string]any)
		node["children"] = append(children, span)
	}

	return traces
}

func canParseRFC3339NanoTime(v string) bool {
	_, err := time.Parse(time.RFC3339Nano, v)
	return err == nil
}

func deleteNilFromMap[M ~map[K]V, K comparable, V any](m M) {
	maps.DeleteFunc(m, func(_ K, value V) bool {
		v := reflect.ValueOf(value)
		switch v.Kind() {
		case reflect.Invalid:
			return true
		case reflect.Interface, reflect.Slice, reflect.Map, reflect.Pointer:
			return v.IsNil()
		default:
			return false
		}
	})
}
