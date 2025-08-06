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
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/nlpodyssey/openai-agents-go/tracing"
)

// VoicePipeline is an opinionated voice agent pipeline.
//
// It works in three steps:
//  1. Transcribe audio input into text.
//  2. Run the provided `workflow`, which produces a sequence of text responses.
//  3. Convert the text responses into streaming audio output.
type VoicePipeline struct {
	workflow VoiceWorkflowBase
	sttModel STTModel
	ttsModel TTSModel
	config   VoicePipelineConfig
}

type VoicePipelineParams struct {
	// The workflow to run.
	Workflow VoiceWorkflowBase

	// Optional speech-to-text model to use.
	// Mutually exclusive with STTModelName.
	// If not provided, a default OpenAI model will be used.
	STTModel STTModel

	// Optional speech-to-text model name.
	// Mutually exclusive with STTModel.
	STTModelName string

	// Optional text-to-speech model to use.
	// Mutually exclusive with TTSModelName.
	// If not provided, a default OpenAI model will be used.
	TTSModel TTSModel

	// Optional text-to-speech model name.
	// Mutually exclusive with TTSModel.
	TTSModelName string

	// Optional pipeline configuration.
	// If not provided, a default configuration will be used.
	Config VoicePipelineConfig
}

// NewVoicePipeline creates a new voice pipeline.
func NewVoicePipeline(params VoicePipelineParams) (*VoicePipeline, error) {
	modelProvider := params.Config.ModelProvider
	if modelProvider == nil {
		modelProvider = NewDefaultOpenAIVoiceModelProvider()
	}

	var err error
	ttsModel := params.TTSModel
	if ttsModel == nil {
		ttsModel, err = modelProvider.GetTTSModel(params.TTSModelName)
		if err != nil {
			return nil, err
		}
	}

	sttModel := params.STTModel
	if sttModel == nil {
		sttModel, err = modelProvider.GetSTTModel(params.STTModelName)
		if err != nil {
			return nil, err
		}
	}

	return &VoicePipeline{
		workflow: params.Workflow,
		sttModel: sttModel,
		ttsModel: ttsModel,
		config:   params.Config,
	}, nil
}

type VoicePipelineAudioInput interface {
	isVoicePipelineInput()
}

func (AudioInput) isVoicePipelineInput()         {}
func (StreamedAudioInput) isVoicePipelineInput() {}

// Run the voice pipeline.
//
// It accepts the audio input to process. This can either be an AudioInput instance,
// which is a single static buffer, or a StreamedAudioInput instance, which is a
// stream of audio data that you can append to.
//
// It returns a StreamedAudioResult instance. You can use this object to stream
// audio events and play them out.
func (p *VoicePipeline) Run(ctx context.Context, audioInput VoicePipelineAudioInput) (*StreamedAudioResult, error) {
	switch audioInput := audioInput.(type) {
	case AudioInput:
		return p.runSingleTurn(ctx, audioInput)
	case StreamedAudioInput:
		return p.runMultiTurn(ctx, audioInput)
	default:
		// This would be an unrecoverable implementation bug, so a panic is appropriate.
		panic(fmt.Errorf("unexpected VoicePipelineAudioInput type %T", audioInput))
	}
}

func (p *VoicePipeline) processAudioInput(ctx context.Context, audionInput AudioInput) (string, error) {
	return p.sttModel.Transcribe(ctx, STTModelTranscribeParams{
		Input:                          audionInput,
		Settings:                       p.config.STTSettings,
		TraceIncludeSensitiveData:      p.config.TraceIncludeSensitiveData.Or(true),
		TraceIncludeSensitiveAudioData: p.config.TraceIncludeSensitiveAudioData.Or(true),
	})
}

func (p *VoicePipeline) runSingleTurn(ctx context.Context, audioInput AudioInput) (*StreamedAudioResult, error) {
	var output *StreamedAudioResult

	// Since this is single turn, we can use the ManageTraceCtx to manage starting/ending the trace
	err := ManageTraceCtx(
		ctx,
		tracing.TraceParams{
			WorkflowName: cmp.Or(p.config.WorkflowName, "Voice Agent"),
			TraceID:      "", // Automatically generated
			GroupID:      p.config.GroupID,
			Metadata:     p.config.TraceMetadata,
			Disabled:     p.config.TracingDisabled,
		},
		func(ctx context.Context) error {
			inputText, err := p.processAudioInput(ctx, audioInput)
			if err != nil {
				return err
			}

			output = NewStreamedAudioResult(p.ttsModel, p.config.TTSSettings, p.config)
			output.createTextGenerationTask(ctx, func(ctx context.Context) (err error) {
				defer func() {
					if err != nil {
						Logger().Error("Error processing single turn", slog.String("error", err.Error()))
						output.addError(err)
					}
				}()
				runResult := p.workflow.Run(ctx, inputText)
				for textEvent := range runResult.Seq() {
					if err = output.addText(ctx, textEvent); err != nil {
						return fmt.Errorf("error adding text to output: %w", err)
					}
				}
				if err = runResult.Error(); err != nil {
					return fmt.Errorf("workflow run error: %w", err)
				}
				output.turnDone(ctx)
				output.done()
				return nil
			})
			return nil
		},
	)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func (p *VoicePipeline) runMultiTurn(ctx context.Context, audioInput StreamedAudioInput) (*StreamedAudioResult, error) {
	var output *StreamedAudioResult

	err := ManageTraceCtx(
		ctx,
		tracing.TraceParams{
			WorkflowName: cmp.Or(p.config.WorkflowName, "Voice Agent"),
			TraceID:      "", // Automatically generated
			GroupID:      p.config.GroupID,
			Metadata:     p.config.TraceMetadata,
			Disabled:     p.config.TracingDisabled,
		},
		func(ctx context.Context) error {
			output = NewStreamedAudioResult(p.ttsModel, p.config.TTSSettings, p.config)

			onStartResult := p.workflow.OnStart(ctx)
			for introText := range onStartResult.Seq() {
				if err := output.addText(ctx, introText); err != nil {
					return fmt.Errorf("error adding text to output: %w", err)
				}
			}
			if err := onStartResult.Error(); err != nil {
				Logger().Error("OnStart() failed", slog.String("error", err.Error()))
			}

			transcriptionSession, err := p.sttModel.CreateSession(ctx, STTModelCreateSessionParams{
				Input:                          audioInput,
				Settings:                       p.config.STTSettings,
				TraceIncludeSensitiveData:      p.config.TraceIncludeSensitiveData.Or(true),
				TraceIncludeSensitiveAudioData: p.config.TraceIncludeSensitiveAudioData.Or(true),
			})
			if err != nil {
				return fmt.Errorf("error creating STT model sesison: %w", err)
			}

			output.createTextGenerationTask(ctx, func(ctx context.Context) (err error) {
				defer func() {
					if err != nil {
						Logger().Error("Error processing turns", slog.String("error", err.Error()))
						output.addError(err)
					}
					if e := transcriptionSession.Close(ctx); e != nil {
						err = errors.Join(err, fmt.Errorf("error closing transcription session: %w", e))
					}
					output.done()
				}()

				tt := transcriptionSession.TranscribeTurns(ctx)

				for inputText := range tt.Seq() {
					result := p.workflow.Run(ctx, inputText)
					for textEvent := range result.Seq() {
						if err = output.addText(ctx, textEvent); err != nil {
							return fmt.Errorf("error adding text to output: %w", err)
						}
					}
					if err = result.Error(); err != nil {
						return fmt.Errorf("workflow run error: %w", err)
					}

					output.turnDone(ctx)
				}
				if err = tt.Error(); err != nil {
					return fmt.Errorf("error transcribing turns: %w", err)
				}

				return nil
			})
			return nil
		},
	)
	if err != nil {
		return nil, err
	}
	return output, nil
}
