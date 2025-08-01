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

// VoiceStreamEvent is an event from the VoicePipeline, streamed via StreamedAudioResult.Stream.
type VoiceStreamEvent interface {
	isVoiceStreamEvent()
}

// VoiceStreamEventAudio is a streaming event from the VoicePipeline.
type VoiceStreamEventAudio struct {
	// The audio data (can be nil).
	Data AudioData
}

func (VoiceStreamEventAudio) isVoiceStreamEvent() {}

type VoiceStreamEventLifecycleEvent string

const (
	VoiceStreamEventLifecycleEventTurnStarted  VoiceStreamEventLifecycleEvent = "turn_started"
	VoiceStreamEventLifecycleEventTurnEnded    VoiceStreamEventLifecycleEvent = "turn_ended"
	VoiceStreamEventLifecycleEventSessionEnded VoiceStreamEventLifecycleEvent = "session_ended"
)

// VoiceStreamEventLifecycle is a streaming event from the VoicePipeline.
type VoiceStreamEventLifecycle struct {
	// The event that occurred.
	Event VoiceStreamEventLifecycleEvent
}

func (VoiceStreamEventLifecycle) isVoiceStreamEvent() {}

// VoiceStreamEventError is a streaming event from the VoicePipeline.
type VoiceStreamEventError struct {
	// The error that occurred.
	Error error
}

func (VoiceStreamEventError) isVoiceStreamEvent() {}
