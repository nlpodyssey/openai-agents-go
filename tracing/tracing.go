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

// AddTraceProcessor adds a new trace processor.
// This processor will receive all traces/spans.
func AddTraceProcessor(spanProcessor Processor) {
	GetTraceProvider().RegisterProcessor(spanProcessor)
}

// SetTraceProcessors sets the list of trace processors.
// This will replace the current list of processors.
func SetTraceProcessors(processors []Processor) {
	GetTraceProvider().SetProcessors(processors)
}

// SetTracingDisabled sets whether tracing is globally disabled.
func SetTracingDisabled(disabled bool) {
	GetTraceProvider().SetDisabled(disabled)
}

// SetTracingExportAPIKey sets the OpenAI API key for the backend exporter.
func SetTracingExportAPIKey(apiKey string) {
	DefaultExporter().SetAPIKey(apiKey)
}

func init() {
	SetTraceProvider(NewDefaultTraceProvider())
	// Add the default processor, which exports traces and spans to the backend in batches.
	// You can change the default behavior by either:
	//  1. calling AddTraceProcessor(), which adds additional processors, or
	//  2. calling SetTraceProcessors(), which replaces the default processor.
	AddTraceProcessor(DefaultProcessor())
}
