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
)

// A Trace is the root level object that tracing creates. It represents a logical "workflow".
type Trace interface {
	// Run calls the given function with this Trace.
	// It allows the implementation of custom actions before/after using the trace.
	Run(context.Context, func(context.Context, Trace) error) error

	// Start the trace.
	// If markAsCurrent is true, the trace will be marked as the current trace.
	Start(ctx context.Context, markAsCurrent bool) error

	// Finish the trace.
	// If resetCurrent is true, the trace will be reset as the current trace.
	Finish(ctx context.Context, resetCurrent bool) error

	// TraceID returns the trace ID.
	TraceID() string

	// The Name of the workflow being traced.
	Name() string

	// Export the trace as a dictionary.
	Export() map[string]any
}

// NoOpTrace is a no-op trace that will not be recorded.
type NoOpTrace struct {
	started          bool
	prevContextTrace Trace
}

func NewNoOpTrace() *NoOpTrace {
	return &NoOpTrace{
		started:          false,
		prevContextTrace: nil,
	}
}

func (t *NoOpTrace) Run(ctx context.Context, fn func(context.Context, Trace) error) (err error) {
	ctx = ContextWithClonedOrNewScope(ctx)

	if t.started {
		if t.prevContextTrace == nil {
			Logger().Error("Trace already started but no context token set")
		}
	} else {
		t.started = true
		err = t.Start(ctx, true)
		if err != nil {
			return err
		}
	}

	defer func() {
		if e := t.Finish(ctx, true); err != nil {
			err = errors.Join(err, e)
		}
	}()

	return fn(ctx, t)
}

func (t *NoOpTrace) Start(ctx context.Context, markAsCurrent bool) error {
	if markAsCurrent {
		t.prevContextTrace = SetCurrentTraceToContextScope(ctx, t)
	}
	return nil
}

func (t *NoOpTrace) Finish(ctx context.Context, resetCurrent bool) error {
	if resetCurrent && t.prevContextTrace != nil {
		SetCurrentTraceToContextScope(ctx, t.prevContextTrace)
		t.prevContextTrace = nil
	}
	return nil
}

func (t *NoOpTrace) TraceID() string        { return "no-op" }
func (t *NoOpTrace) Name() string           { return "no-op" }
func (t *NoOpTrace) Export() map[string]any { return nil }

// TraceImpl is a trace that will be recorded by the tracing library.
type TraceImpl struct {
	name             string
	traceID          string
	GroupID          string
	Metadata         map[string]any
	processor        Processor
	prevContextTrace Trace
	started          bool
}

func NewTraceImpl(
	name string,
	traceID string,
	groupID string,
	metadata map[string]any,
	processor Processor,
) *TraceImpl {
	if traceID == "" {
		traceID = GenTraceID()
	}
	return &TraceImpl{
		name:             name,
		traceID:          traceID,
		GroupID:          groupID,
		Metadata:         metadata,
		processor:        processor,
		prevContextTrace: nil,
		started:          false,
	}
}

func (t *TraceImpl) Run(ctx context.Context, fn func(context.Context, Trace) error) (err error) {
	ctx = ContextWithClonedOrNewScope(ctx)

	if t.started {
		if t.prevContextTrace == nil {
			Logger().Error("Trace already started but no context token set")
		}
	} else {
		err = t.Start(ctx, true)
		if err != nil {
			return err
		}
	}

	defer func() {
		if e := t.Finish(ctx, true); err != nil {
			err = errors.Join(err, e)
		}
	}()

	return fn(ctx, t)
}

func (t *TraceImpl) Start(ctx context.Context, markAsCurrent bool) error {
	if t.started {
		return nil
	}

	t.started = true
	err := t.processor.OnTraceStart(ctx, t)
	if err != nil {
		return err
	}

	if markAsCurrent {
		t.prevContextTrace = SetCurrentTraceToContextScope(ctx, t)
	}
	return nil
}

func (t *TraceImpl) Finish(ctx context.Context, resetCurrent bool) error {
	if !t.started {
		return nil
	}

	err := t.processor.OnTraceEnd(ctx, t)
	if err != nil {
		return err
	}

	if resetCurrent && t.prevContextTrace != nil {
		SetCurrentTraceToContextScope(ctx, t.prevContextTrace)
		t.prevContextTrace = nil
	}
	return nil
}

func (t *TraceImpl) TraceID() string { return t.traceID }
func (t *TraceImpl) Name() string    { return t.name }

func (t *TraceImpl) Export() map[string]any {
	var groupID any
	if t.GroupID != "" {
		groupID = t.GroupID
	}
	return map[string]any{
		"object":        "trace",
		"id":            t.TraceID(),
		"workflow_name": t.Name(),
		"group_id":      groupID,
		"metadata":      t.Metadata,
	}
}
