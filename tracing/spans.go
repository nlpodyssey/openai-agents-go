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
	"cmp"
	"context"
	"errors"
	"sync/atomic"
)

//TSpanData = TypeVar("TSpanData", bound=SpanData)
//

type SpanError struct {
	Message string
	Data    map[string]any
}

func (err SpanError) Error() string { return cmp.Or(err.Message, "span error") }

func (err SpanError) Export() map[string]any {
	return map[string]any{
		"message": err.Message,
		"data":    err.Data,
	}
}

type Span interface {
	// Run calls the given function with this Span.
	// It allows the implementation of custom actions before/after using the span.
	Run(context.Context, func(context.Context, Span) error) error

	// Start the span.SpanData
	// If markAsCurrent is true, the span will be marked as the current span.
	Start(ctx context.Context, markAsCurrent bool) error

	// Finish the span.
	// If resetCurrent is true, the span will be reset as the current span.
	Finish(ctx context.Context, resetCurrent bool) error

	TraceID() string
	SpanID() string
	SpanData() SpanData
	ParentID() string
	SetError(err SpanError)
	Error() *SpanError
	StartedAt() string
	EndedAt() string
	Export() map[string]any
}

type NoOpSpan struct {
	spanData        SpanData
	prevContextSpan *Span
}

func NewNoOpSpan(spanData SpanData) *NoOpSpan {
	return &NoOpSpan{
		spanData:        spanData,
		prevContextSpan: nil,
	}
}

func (s *NoOpSpan) Run(ctx context.Context, fn func(context.Context, Span) error) (err error) {
	ctx = ContextWithClonedOrNewScope(ctx)

	err = s.Start(ctx, true)
	if err != nil {
		return err
	}

	defer func() {
		if e := s.Finish(ctx, true); err != nil {
			err = errors.Join(err, e)
		}
	}()

	return fn(ctx, s)
}

func (s *NoOpSpan) Start(ctx context.Context, markAsCurrent bool) error {
	if markAsCurrent {
		prevSpan := SetCurrentSpanToContextScope(ctx, s)
		s.prevContextSpan = &prevSpan
	}
	return nil
}

func (s *NoOpSpan) Finish(ctx context.Context, resetCurrent bool) error {
	if resetCurrent && s.prevContextSpan != nil {
		SetCurrentSpanToContextScope(ctx, *s.prevContextSpan)
		s.prevContextSpan = nil
	}
	return nil
}

func (s *NoOpSpan) TraceID() string        { return "no-op" }
func (s *NoOpSpan) SpanID() string         { return "no-op" }
func (s *NoOpSpan) SpanData() SpanData     { return s.spanData }
func (s *NoOpSpan) ParentID() string       { return "" }
func (s *NoOpSpan) SetError(SpanError)     {}
func (s *NoOpSpan) Error() *SpanError      { return nil }
func (s *NoOpSpan) StartedAt() string      { return "" }
func (s *NoOpSpan) EndedAt() string        { return "" }
func (s *NoOpSpan) Export() map[string]any { return nil }

type SpanImpl struct {
	traceID         string
	spanID          string
	parentID        string
	startedAt       string
	endedAt         string
	error           atomic.Pointer[SpanError]
	prevContextSpan *Span
	processor       Processor
	spanData        SpanData
}

func NewSpanImpl(
	traceID string,
	spanID string,
	parentID string,
	processor Processor,
	spanData SpanData,
) *SpanImpl {
	if spanID == "" {
		spanID = GenSpanID()
	}
	return &SpanImpl{
		traceID:         traceID,
		spanID:          spanID,
		parentID:        parentID,
		startedAt:       "",
		endedAt:         "",
		prevContextSpan: nil,
		processor:       processor,
		spanData:        spanData,
	}
}

func (s *SpanImpl) Run(ctx context.Context, fn func(context.Context, Span) error) (err error) {
	ctx = ContextWithClonedOrNewScope(ctx)

	err = s.Start(ctx, true)
	if err != nil {
		return err
	}

	defer func() {
		if e := s.Finish(ctx, true); err != nil {
			err = errors.Join(err, e)
		}
	}()

	return fn(ctx, s)
}

func (s *SpanImpl) Start(ctx context.Context, markAsCurrent bool) error {
	if s.startedAt != "" {
		Logger().Warn("Span already started")
		return nil
	}

	s.startedAt = TimeISO()
	err := s.processor.OnSpanStart(ctx, s)
	if err != nil {
		return err
	}

	if markAsCurrent {
		prevSpan := SetCurrentSpanToContextScope(ctx, s)
		s.prevContextSpan = &prevSpan
	}
	return nil
}

func (s *SpanImpl) Finish(ctx context.Context, resetCurrent bool) error {
	if s.endedAt != "" {
		Logger().Warn("Span already finished")
		return nil
	}

	s.endedAt = TimeISO()
	err := s.processor.OnSpanEnd(ctx, s)
	if err != nil {
		return err
	}

	if resetCurrent && s.prevContextSpan != nil {
		SetCurrentSpanToContextScope(ctx, *s.prevContextSpan)
		s.prevContextSpan = nil
	}
	return nil
}

func (s *SpanImpl) TraceID() string        { return s.traceID }
func (s *SpanImpl) SpanID() string         { return s.spanID }
func (s *SpanImpl) SpanData() SpanData     { return s.spanData }
func (s *SpanImpl) ParentID() string       { return s.parentID }
func (s *SpanImpl) SetError(err SpanError) { s.error.Store(&err) }
func (s *SpanImpl) Error() *SpanError      { return s.error.Load() }
func (s *SpanImpl) StartedAt() string      { return s.startedAt }
func (s *SpanImpl) EndedAt() string        { return s.endedAt }

func (s *SpanImpl) Export() map[string]any {
	var spanData map[string]any
	if s.SpanData() != nil {
		spanData = s.SpanData().Export()
	}

	var exportedError map[string]any
	if err := s.error.Load(); err != nil {
		exportedError = err.Export()
	}

	var parentID any
	if s.parentID != "" {
		parentID = s.parentID
	}
	var startedAt any
	if s.startedAt != "" {
		startedAt = s.startedAt
	}
	var endedAt any
	if s.endedAt != "" {
		endedAt = s.endedAt
	}

	return map[string]any{
		"object":     "trace.span",
		"id":         s.SpanID(),
		"trace_id":   s.TraceID(),
		"parent_id":  parentID,
		"started_at": startedAt,
		"ended_at":   endedAt,
		"span_data":  spanData,
		"error":      exportedError,
	}
}
