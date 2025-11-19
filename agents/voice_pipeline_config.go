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

package agents

import "github.com/openai/openai-go/v3/packages/param"

// VoicePipelineConfig provides configuration settings for a VoicePipeline.
type VoicePipelineConfig struct {
	// Optional voice model provider to use for the pipeline.
	// Defaults to OpenAIVoiceModelProvider.
	ModelProvider VoiceModelProvider

	// Whether to disable tracing of the pipeline. Defaults to false.
	TracingDisabled bool

	// Whether to include sensitive data in traces. Defaults to true. This is specifically for the
	//  voice pipeline, and not for anything that goes on inside your Workflow.
	TraceIncludeSensitiveData param.Opt[bool]

	// Whether to include audio data in traces. Defaults to true.
	TraceIncludeSensitiveAudioData param.Opt[bool]

	// Optional name of the workflow to use for tracing. Defaults to "Voice Agent".
	WorkflowName string

	// Optional grouping identifier to use for tracing, to link multiple traces from the same conversation
	// or process. If not provided, we will create a random group ID with tracing.GenGroupID.
	GroupID string

	// An optional dictionary of additional metadata to include with the trace.
	TraceMetadata map[string]any

	// The settings to use for the STT model.
	STTSettings STTModelSettings

	// The settings to use for the TTS model.
	TTSSettings TTSModelSettings
}
