package agents

import (
	"context"
	"log/slog"

	"github.com/nlpodyssey/openai-agents-go/tracing"
)

func AttachErrorToSpan(span tracing.Span, err tracing.SpanError) {
	span.SetError(err)
}

func AttachErrorToCurrentSpan(ctx context.Context, err tracing.SpanError) {
	span := tracing.GetCurrentSpan(ctx)
	if span != nil {
		AttachErrorToSpan(span, err)
	} else {
		Logger().Warn("No span to add error to", slog.String("error", err.Error()))
	}
}
