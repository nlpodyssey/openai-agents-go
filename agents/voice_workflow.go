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

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
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
	return &VoiceWorkflowHelperStreamTextFromResult{
		result: result,
		err:    nil,
	}
}

type VoiceWorkflowHelperStreamTextFromResult struct {
	result *RunResultStreaming
	err    error
}

func (r *VoiceWorkflowHelperStreamTextFromResult) Seq() iter.Seq[string] {
	return func(yield func(string) bool) {
		canYield := true // once yield returns false, stop yielding, but finish consuming all events
		r.err = r.result.StreamEvents(func(event StreamEvent) error {
			if !canYield {
				return nil
			}
			if e, ok := event.(RawResponsesStreamEvent); ok && e.Data.Type == "response.output_text.delta" {
				canYield = yield(e.Data.Delta)
			}
			return nil
		})
	}
}

func (r *VoiceWorkflowHelperStreamTextFromResult) Error() error {
	return r.err
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
	return &singleAgentVoiceWorkflowRunResult{
		ctx:           ctx,
		workflow:      w,
		transcription: transcription,
		err:           nil,
	}
}

func (w *SingleAgentVoiceWorkflow) OnStart(context.Context) VoiceWorkflowBaseOnStartResult {
	return NoOpVoiceWorkflowBaseOnStartResult{}
}

type singleAgentVoiceWorkflowRunResult struct {
	ctx           context.Context
	workflow      *SingleAgentVoiceWorkflow
	transcription string
	err           error
}

func (r *singleAgentVoiceWorkflowRunResult) Seq() iter.Seq[string] {
	return func(yield func(string) bool) {
		if r.workflow.callbacks != nil {
			r.err = r.workflow.callbacks.OnRun(r.ctx, r.workflow, r.transcription)
			if r.err != nil {
				return
			}
		}

		// Add the transcription to the input history
		r.workflow.inputHistory = append(r.workflow.inputHistory, TResponseInputItem{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt(r.transcription),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		})

		// Run the agent
		result, err := RunInputsStreamed(r.ctx, r.workflow.currentAgent, r.workflow.inputHistory)
		if err != nil {
			r.err = err
			return
		}

		// Stream the text from the result
		textStream := VoiceWorkflowHelper().StreamTextFrom(result)
		for chunk := range textStream.Seq() {
			if !yield(chunk) {
				break
			}
		}
		r.err = textStream.Error()
		if r.err != nil {
			return
		}

		// Update the input history and current agent
		r.workflow.inputHistory = result.ToInputList()
		r.workflow.currentAgent = result.LastAgent()
	}
}

func (r *singleAgentVoiceWorkflowRunResult) Error() error {
	return r.err
}
