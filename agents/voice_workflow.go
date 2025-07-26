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

import (
	"context"
	"iter"
	"sync/atomic"

	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

// VoiceWorkflowBase is the base interface for a voice workflow.
//
// A "workflow" is any code you want, that receives a transcription and yields
// text that will be turned into speech by a text-to-speech model.
// In most cases, you'll create Agents and use RunStreamed to run them, returning
// some or all of the text events from the stream.
// You can use VoiceWorkflowHelper to help with extracting text events from the stream.
// If you have a simple workflow that has a single starting agent and no custom logic, you can
// use SingleAgentVoiceWorkflow directly.
type VoiceWorkflowBase interface {
	// Run the voice workflow. You will receive an input transcription, and must yield text that
	// will be spoken to the user. You can run whatever logic you want here. In most cases, the
	// final logic will involve calling RunStreamed and yielding any text events from
	// the stream.
	Run(ctx context.Context, transcription string) VoiceWorkflowBaseRunResult

	// OnStart runs before any user input is received. It can be used
	// to deliver a greeting or instruction via TTS.
	OnStart(context.Context) VoiceWorkflowBaseOnStartResult
}

type VoiceWorkflowBaseRunResult interface {
	Seq() iter.Seq[string]
	Error() error
}

type VoiceWorkflowBaseOnStartResult interface {
	Seq() iter.Seq[string]
	Error() error
}

type NoOpVoiceWorkflowBaseOnStartResult struct{}

func (n NoOpVoiceWorkflowBaseOnStartResult) Seq() iter.Seq[string] { return func(func(string) bool) {} }
func (n NoOpVoiceWorkflowBaseOnStartResult) Error() error          { return nil }

var _ VoiceWorkflowBaseOnStartResult = NoOpVoiceWorkflowBaseOnStartResult{}

type voiceWorkflowHelper struct{}

func VoiceWorkflowHelper() voiceWorkflowHelper { return voiceWorkflowHelper{} }

// StreamTextFrom wraps a RunResultStreaming object and yields text events from the stream.
func (voiceWorkflowHelper) StreamTextFrom(result *RunResultStreaming) *VoiceWorkflowHelperStreamTextFromResult {
	r := &VoiceWorkflowHelperStreamTextFromResult{ch: make(chan string)}
	go func() {
		defer close(r.ch)

		err := result.StreamEvents(func(event StreamEvent) error {
			if e, ok := event.(RawResponsesStreamEvent); ok && e.Data.Type == "response.output_text.delta" {
				r.ch <- e.Data.Delta.OfString
			}
			return nil
		})
		if err != nil {
			r.err.Store(&err)
		}
	}()
	return r
}

type VoiceWorkflowHelperStreamTextFromResult struct {
	ch  chan string
	err atomic.Pointer[error]
}

func (r *VoiceWorkflowHelperStreamTextFromResult) Seq() iter.Seq[string] {
	return func(yield func(string) bool) {
		for v := range r.ch {
			if !yield(v) {
				return
			}
		}
	}
}

func (r *VoiceWorkflowHelperStreamTextFromResult) Error() error {
	p := r.err.Load()
	if p != nil {
		return *p
	}
	return nil
}

type SingleAgentWorkflowCallbacks interface {
	// OnRun is Called when the workflow is run.
	OnRun(ctx context.Context, workflow *SingleAgentVoiceWorkflow, transcription string) error
}

// SingleAgentVoiceWorkflow is a simple voice workflow that runs a single agent.
// Each transcription and result is added to the input history.
// For more complex workflows (e.g. multiple runner calls, custom message history, custom logic,
// custom configs), implement a VoiceWorkflowBase with your own logic.
type SingleAgentVoiceWorkflow struct {
	inputHistory []TResponseInputItem
	currentAgent *Agent
	callbacks    SingleAgentWorkflowCallbacks
}

// NewSingleAgentVoiceWorkflow creates a new single agent voice workflow.
func NewSingleAgentVoiceWorkflow(agent *Agent, callbacks SingleAgentWorkflowCallbacks) *SingleAgentVoiceWorkflow {
	return &SingleAgentVoiceWorkflow{
		inputHistory: nil,
		currentAgent: agent,
		callbacks:    callbacks,
	}
}

func (w *SingleAgentVoiceWorkflow) Run(ctx context.Context, transcription string) VoiceWorkflowBaseRunResult {
	r := &singleAgentVoiceWorkflowRunResult{ch: make(chan string)}
	go func() {
		defer close(r.ch)

		if w.callbacks != nil {
			err := w.callbacks.OnRun(ctx, w, transcription)
			if err != nil {
				r.err.Store(&err)
				return
			}
		}

		// Add the transcription to the input history
		w.inputHistory = append(w.inputHistory, TResponseInputItem{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt(transcription),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		})

		// Run the agent
		result, err := RunInputsStreamed(ctx, w.currentAgent, w.inputHistory)
		if err != nil {
			r.err.Store(&err)
			return
		}

		// Stream the text from the result
		textStream := VoiceWorkflowHelper().StreamTextFrom(result)
		for chunk := range textStream.Seq() {
			r.ch <- chunk
		}
		if err = textStream.Error(); err != nil {
			r.err.Store(&err)
			return
		}

		// Update the input history and current agent
		w.inputHistory = result.ToInputList()
		w.currentAgent = result.LastAgent()
	}()
	return r
}

func (w *SingleAgentVoiceWorkflow) OnStart(context.Context) VoiceWorkflowBaseOnStartResult {
	return NoOpVoiceWorkflowBaseOnStartResult{}
}

type singleAgentVoiceWorkflowRunResult struct {
	ch  chan string
	err atomic.Pointer[error]
}

func (r *singleAgentVoiceWorkflowRunResult) Seq() iter.Seq[string] {
	return func(yield func(string) bool) {
		for v := range r.ch {
			if !yield(v) {
				return
			}
		}
	}
}

func (r *singleAgentVoiceWorkflowRunResult) Error() error {
	p := r.err.Load()
	if p != nil {
		return *p
	}
	return nil
}
