package tracingtesting

import (
	"context"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/tracing"
)

func Setup(t *testing.T) {
	SetupCtx(t, t.Context())
}

func SetupCtx(t *testing.T, ctx context.Context) {
	t.Helper()
	SetupSpanProcessor()
	ClearSpanProcessor(t, ctx)

	t.Cleanup(func() {
		ShutdownTraceProvider(ctx)
	})
}

func SetupSpanProcessor() {
	tracing.SetTraceProcessors([]tracing.Processor{SpanProcessorTesting()})
}

func ClearSpanProcessor(t *testing.T, ctx context.Context) {
	t.Helper()
	if err := SpanProcessorTesting().ForceFlush(ctx); err != nil {
		t.Fatal(err)
	}
	if err := SpanProcessorTesting().Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	SpanProcessorTesting().Clear()
}

func ShutdownTraceProvider(ctx context.Context) {
	tracing.GetTraceProvider().Shutdown(ctx)
}
