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
	"sync"
)

type scopeContextKey struct{}

type Scope struct {
	currentSpan  Span
	currentTrace Trace
	mu           sync.RWMutex
}

func (s *Scope) clone() *Scope {
	return &Scope{
		currentSpan:  s.currentSpan,
		currentTrace: s.currentTrace,
	}
}

// ContextWithClonedOrNewScope returns a context derived from ctx with a Scope value set.
// If a Scope is already present in the given context, the new context will contain its clone
// (new instance, same values), otherwise a new Scope is created and set.
func ContextWithClonedOrNewScope(ctx context.Context) context.Context {
	if scope, ok := ScopeFromContext(ctx); ok {
		return context.WithValue(ctx, scopeContextKey{}, scope.clone())
	}
	return context.WithValue(ctx, scopeContextKey{}, new(Scope))
}

func ScopeFromContext(ctx context.Context) (*Scope, bool) {
	scope, ok := ctx.Value(scopeContextKey{}).(*Scope)
	return scope, ok
}

func GetCurrentSpanFromContextScope(ctx context.Context) Span {
	scope, ok := ScopeFromContext(ctx)
	if !ok {
		return nil
	}
	scope.mu.RLock()
	defer scope.mu.RUnlock()
	return scope.currentSpan
}

func SetCurrentSpanToContextScope(ctx context.Context, span Span) (previousSpan Span) {
	scope, ok := ScopeFromContext(ctx)
	if ok {
		scope.mu.Lock()
		defer scope.mu.Unlock()
		previousSpan = scope.currentSpan
		scope.currentSpan = span
	}
	return previousSpan
}

func GetCurrentTraceFromContextScope(ctx context.Context) Trace {
	scope, ok := ScopeFromContext(ctx)
	if !ok {
		return nil
	}
	scope.mu.RLock()
	defer scope.mu.RUnlock()
	return scope.currentTrace
}

func SetCurrentTraceToContextScope(ctx context.Context, trace Trace) (previousTrace Trace) {
	scope, ok := ScopeFromContext(ctx)
	if ok {
		scope.mu.Lock()
		defer scope.mu.Unlock()
		previousTrace = scope.currentTrace
		scope.currentTrace = trace
	}
	return previousTrace
}
