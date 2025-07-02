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

import "context"

// Processor is an interface for processing spans.
type Processor interface {
	// OnTraceStart is called when a trace is started.
	OnTraceStart(context.Context, Trace) error

	// OnTraceEnd is called when a trace is finished.
	OnTraceEnd(context.Context, Trace) error

	// OnSpanStart is called when a span is started.
	OnSpanStart(context.Context, Span) error

	// OnSpanEnd is called when a span is finished.
	OnSpanEnd(context.Context, Span) error

	// Shutdown is called when the application stops.
	Shutdown(context.Context) error

	// ForceFlush forces an immediate flush of all queued spans/traces.
	ForceFlush(ctx context.Context) error
}

// Exporter is implemented by objects which export traces and spans.
// For example, could log them or send them to a backend.
type Exporter interface {
	// Export a list of traces and spans. Each item is either a Trace or a Span.
	Export(ctx context.Context, items []any) error
}
