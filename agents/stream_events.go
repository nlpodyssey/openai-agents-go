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

// StreamEvent is a streaming event from an agent.
type StreamEvent interface {
	isStreamEvent()
}

// RawResponsesStreamEvent is a streaming event from the LLM.
// These are 'raw' events, i.e. they are directly passed through from the LLM.
type RawResponsesStreamEvent struct {
	// The raw responses streaming event from the LLM.
	Data TResponseStreamEvent

	// The type of the event. Always `raw_response_event`.
	Type string
}

func (RawResponsesStreamEvent) isStreamEvent() {}

// RunItemStreamEvent is a streaming event that wrap a `RunItem`.
// As the agent processes the LLM response, it will generate these events for
// new messages, tool calls, tool outputs, handoffs, etc.
type RunItemStreamEvent struct {
	// The name of the event.
	Name RunItemStreamEventName

	// The item that was created.
	Item RunItem

	// Always `run_item_stream_event`.
	Type string
}

func (RunItemStreamEvent) isStreamEvent() {}

func NewRunItemStreamEvent(name RunItemStreamEventName, item RunItem) RunItemStreamEvent {
	return RunItemStreamEvent{
		Name: name,
		Item: item,
		Type: "run_item_stream_event",
	}
}

type RunItemStreamEventName string

const (
	StreamEventMessageOutputCreated RunItemStreamEventName = "message_output_created"
	StreamEventHandoffRequested     RunItemStreamEventName = "handoff_requested"
	StreamEventHandoffOccurred      RunItemStreamEventName = "handoff_occurred"
	StreamEventToolCalled           RunItemStreamEventName = "tool_called"
	StreamEventToolOutput           RunItemStreamEventName = "tool_output"
	StreamEventReasoningItemCreated RunItemStreamEventName = "reasoning_item_created"
)

// AgentUpdatedStreamEvent is an event that notifies that there is a new agent running.
type AgentUpdatedStreamEvent struct {
	// The new agent.
	NewAgent *Agent

	// Always `agent_updated_stream_event`.
	Type string
}

func (AgentUpdatedStreamEvent) isStreamEvent() {}
