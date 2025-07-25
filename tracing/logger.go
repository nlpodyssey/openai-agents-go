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
	"log/slog"
	"os"
	"sync/atomic"
)

var tracingLogger atomic.Pointer[slog.Logger]

func init() {
	opts := &slog.HandlerOptions{Level: slog.LevelInfo}
	tracingLogger.Store(slog.New(slog.NewTextHandler(os.Stderr, opts)))
}

// Logger is the global logger used by Agents SDK tracing package.
// By default, it is a logger with a text handler which writes to stdout,
// with minimum level "info". You can change it with SetLogger.
func Logger() *slog.Logger {
	return tracingLogger.Load()
}

// SetLogger sets the global logger use by Agents SDK tracing package.
// A nil value is ignored.
func SetLogger(l *slog.Logger) {
	if l != nil {
		tracingLogger.Store(l)
	}
}
