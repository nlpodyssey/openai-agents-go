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
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/google/uuid"
)

// SynchronousMultiTracingProcessor forwards all calls to a list of Processors, in order of registration.
type SynchronousMultiTracingProcessor struct {
	processors []Processor
	mu         sync.RWMutex
}

func NewSynchronousMultiTracingProcessor() *SynchronousMultiTracingProcessor {
	return &SynchronousMultiTracingProcessor{}
}

// AddProcessor adds a processor to the list of processors.
// Each processor will receive all traces/spans.
func (p *SynchronousMultiTracingProcessor) AddProcessor(processor Processor) {
	p.mu.Lock()
	p.processors = append(p.processors, processor)
	p.mu.Unlock()
}

// SetProcessors sets the list of processors.
// This will replace the current list of processors.
func (p *SynchronousMultiTracingProcessor) SetProcessors(processors []Processor) {
	p.mu.Lock()
	p.processors = slices.Clone(processors)
	p.mu.Unlock()
}

// OnTraceStart is called when a trace is started.
func (p *SynchronousMultiTracingProcessor) OnTraceStart(ctx context.Context, trace Trace) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	errs := make([]error, len(p.processors))
	for i, processor := range p.processors {
		errs[i] = processor.OnTraceStart(ctx, trace)
	}
	return errors.Join(errs...)
}

// OnTraceEnd is called when a trace is finished.
func (p *SynchronousMultiTracingProcessor) OnTraceEnd(ctx context.Context, trace Trace) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	errs := make([]error, len(p.processors))
	for i, processor := range p.processors {
		errs[i] = processor.OnTraceEnd(ctx, trace)
	}
	return errors.Join(errs...)
}

// OnSpanStart is called when a span is started.
func (p *SynchronousMultiTracingProcessor) OnSpanStart(ctx context.Context, span Span) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	errs := make([]error, len(p.processors))
	for i, processor := range p.processors {
		errs[i] = processor.OnSpanStart(ctx, span)
	}
	return errors.Join(errs...)
}

// OnSpanEnd is called when a span is finished.
func (p *SynchronousMultiTracingProcessor) OnSpanEnd(ctx context.Context, span Span) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	errs := make([]error, len(p.processors))
	for i, processor := range p.processors {
		errs[i] = processor.OnSpanEnd(ctx, span)
	}
	return errors.Join(errs...)
}

// Shutdown is called when the application stops.
func (p *SynchronousMultiTracingProcessor) Shutdown(ctx context.Context) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	errs := make([]error, len(p.processors))
	for i, processor := range p.processors {
		errs[i] = processor.Shutdown(ctx)
	}
	return errors.Join(errs...)
}

func (p *SynchronousMultiTracingProcessor) ForceFlush(ctx context.Context) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	errs := make([]error, len(p.processors))
	for i, processor := range p.processors {
		errs[i] = processor.ForceFlush(ctx)
	}
	return errors.Join(errs...)
}

// TraceProvider is an interface for creating traces and spans.
type TraceProvider interface {
	// RegisterProcessor adds a processor that will receive all traces and spans.
	RegisterProcessor(processor Processor)

	// SetProcessors replaces the list of processors with the given value.
	SetProcessors(processors []Processor)

	// GetCurrentTrace returns the currently active trace, if any.
	GetCurrentTrace(context.Context) Trace

	// GetCurrentSpan returns the currently active span, if any.
	GetCurrentSpan(context.Context) Span

	// SetDisabled enable or disable tracing globally.
	SetDisabled(disabled bool)

	// GenTraceID generates a new trace identifier.
	GenTraceID() string

	// GenSpanID generates a new span identifier.
	GenSpanID() string

	// GenGroupID generates a new group identifier.
	GenGroupID() string

	// CreateTrace creates a new trace.
	CreateTrace(
		name string,
		traceID string,
		groupID string,
		metadata map[string]any,
		disabled bool,
	) Trace

	// CreateSpan creates a new span.
	CreateSpan(
		ctx context.Context,
		spanData SpanData,
		spanID string,
		parent any,
		disabled bool,
	) Span

	// Shutdown cleans up any resources used by the provider.
	Shutdown(context.Context)
}

type DefaultTraceProvider struct {
	multiProcessor *SynchronousMultiTracingProcessor
	disabled       bool
}

func NewDefaultTraceProvider() *DefaultTraceProvider {
	disabledVar := strings.ToLower(os.Getenv("OPENAI_AGENTS_DISABLE_TRACING"))

	return &DefaultTraceProvider{
		multiProcessor: NewSynchronousMultiTracingProcessor(),
		disabled:       disabledVar == "true" || disabledVar == "1",
	}
}

// RegisterProcessor adds a processor to the list of processors.
// Each processor will receive all traces/spans.
func (p *DefaultTraceProvider) RegisterProcessor(processor Processor) {
	p.multiProcessor.AddProcessor(processor)
}

// SetProcessors sets the list of processors.
// This will replace the current list of processors.
func (p *DefaultTraceProvider) SetProcessors(processors []Processor) {
	p.multiProcessor.SetProcessors(processors)
}

// GetCurrentTrace returns the currently active trace, if any.
func (p *DefaultTraceProvider) GetCurrentTrace(ctx context.Context) Trace {
	return GetCurrentTraceFromContextScope(ctx)
}

// GetCurrentSpan returns the currently active span, if any.
func (p *DefaultTraceProvider) GetCurrentSpan(ctx context.Context) Span {
	return GetCurrentSpanFromContextScope(ctx)
}

// SetDisabled set whether tracing is disabled.
func (p *DefaultTraceProvider) SetDisabled(disabled bool) {
	p.disabled = disabled
}

// GenTraceID generates a new trace ID.
func (p *DefaultTraceProvider) GenTraceID() string {
	u := uuid.New()
	return "trace_" + hex.EncodeToString(u[:])
}

// GenSpanID generates a new span ID.
func (p *DefaultTraceProvider) GenSpanID() string {
	u := uuid.New()
	return "span_" + hex.EncodeToString(u[:])[:24]
}

// GenGroupID generates a new group ID.
func (p *DefaultTraceProvider) GenGroupID() string {
	u := uuid.New()
	return "group_" + hex.EncodeToString(u[:])[:24]
}

// CreateTrace creates a new trace.
func (p *DefaultTraceProvider) CreateTrace(
	name string,
	traceID string,
	groupID string,
	metadata map[string]any,
	disabled bool,
) Trace {
	if p.disabled || disabled {
		Logger().Debug("Tracing is disabled. Not creating trace", slog.String("name", name))
		return NewNoOpTrace()
	}

	if traceID == "" {
		traceID = p.GenTraceID()
	}

	Logger().Debug("Creating trace", slog.String("name", name), slog.String("ID", traceID))

	return NewTraceImpl(name, traceID, groupID, metadata, p.multiProcessor)
}

// CreateSpan creates a new span.
func (p *DefaultTraceProvider) CreateSpan(
	ctx context.Context,
	spanData SpanData,
	spanID string,
	parent any,
	disabled bool,
) Span {
	if p.disabled || disabled {
		Logger().Debug("Tracing is disabled. Not creating span", slog.Any("data", spanData))
		return NewNoOpSpan(spanData)
	}

	var parentID string
	var traceID string

	switch parent := parent.(type) {
	case nil:
		currentSpan := GetCurrentSpanFromContextScope(ctx)
		currentTrace := GetCurrentTraceFromContextScope(ctx)

		if currentTrace == nil {
			Logger().Error("No active trace. Make sure to start a trace first. Returning NoOpSpan.")
			return NewNoOpSpan(spanData)
		}

		if _, ok := currentTrace.(*NoOpTrace); ok {
			Logger().Error("Current parent trace is no-op. Returning NoOpSpan.")
			return NewNoOpSpan(spanData)
		}
		if _, ok := currentSpan.(*NoOpSpan); ok {
			Logger().Error("Current parent span is no-op. Returning NoOpSpan.")
			return NewNoOpSpan(spanData)
		}

		if currentSpan != nil {
			parentID = currentSpan.SpanID()
		}
		traceID = currentTrace.TraceID()
	case Trace:
		if _, ok := parent.(*NoOpTrace); ok {
			Logger().Debug("Parent trace is no-op, returning NoOpSpan.")
			return NewNoOpSpan(spanData)
		}
		traceID = parent.TraceID()
		parentID = ""
	case Span:
		if _, ok := parent.(*NoOpSpan); ok {
			Logger().Debug("Parent span is no-op, returning NoOpSpan.")
			return NewNoOpSpan(spanData)
		}
		parentID = parent.SpanID()
		traceID = parent.TraceID()
	default:
		Logger().Error(fmt.Sprintf("Unexpected parent type %T. Returning NoOpSpan.", parent))
		return NewNoOpSpan(spanData)
	}

	Logger().Debug("Creating span", slog.Any("data", spanData), slog.String("ID", spanID))

	if spanID == "" {
		spanID = p.GenSpanID()
	}

	return NewSpanImpl(traceID, spanID, parentID, p.multiProcessor, spanData)
}

func (p *DefaultTraceProvider) Shutdown(ctx context.Context) {
	if p.disabled {
		return
	}

	Logger().Debug("Shutting down trace provider")

	err := p.multiProcessor.Shutdown(ctx)
	if err != nil {
		Logger().Error("Error shutting down trace provider", slog.String("error", err.Error()))
	}
}
