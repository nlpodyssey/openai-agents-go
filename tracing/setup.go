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
	"errors"
	"sync/atomic"
)

var globalTraceProvider atomic.Pointer[TraceProvider]

// SetTraceProvider sets the global trace provider used by tracing utilities.
// A nil value is ignored.
func SetTraceProvider(provider TraceProvider) {
	if provider != nil {
		globalTraceProvider.Store(&provider)
	}
}

// GetTraceProvider returns the global trace provider used by tracing utilities.
// It panics if a trace provider is not set.
// Use SafeGetTraceProvider for a safer alternative.
func GetTraceProvider() TraceProvider {
	v, ok := SafeGetTraceProvider()
	if !ok {
		panic(errors.New("trace provider not set"))
	}
	return v
}

// SafeGetTraceProvider returns the global trace provider used by tracing utilities.
func SafeGetTraceProvider() (TraceProvider, bool) {
	v := globalTraceProvider.Load()
	if v == nil || *v == nil {
		return nil, false
	}
	return *v, true
}
