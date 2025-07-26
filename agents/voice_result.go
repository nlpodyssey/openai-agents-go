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
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"time"

	"github.com/nlpodyssey/openai-agents-go/asyncqueue"
	"github.com/nlpodyssey/openai-agents-go/asynctask"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

func audioToBase64(audioData [][]byte) string {
	flatLen := 0
	for _, v := range audioData {
		flatLen += len(v)
	}

	flatData := make([]byte, 0, flatLen)
	for _, v := range audioData {
		flatData = append(flatData, v...)
	}

	return base64.StdEncoding.EncodeToString(flatData)
}

// StreamedAudioResult is the output of a VoicePipeline.
// Streams events and audio data as they're generated.
type StreamedAudioResult struct {
	ttsModel            TTSModel
	ttsSettings         TTSModelSettings
	totalOutputText     string
	instructions        string
	textGenerationTask  *asynctask.Task[any] // TODO: can be TaskNoValue?
	voicePipelineConfig VoicePipelineConfig
	textBuffer          string
	turnTextBuffer      string
	queue               *asyncqueue.Queue[VoiceStreamEvent]
	tasks               []*asynctask.Task[any] // TODO: can be TaskNoValue?
	// List to hold local queues for each text segment
	orderedTasks []*asyncqueue.Queue[VoiceStreamEvent]
	// Task to dispatch audio chunks in order
	dispatcherTask        *asynctask.Task[any] // TODO: can be TaskNoValue?
	doneProcessing        bool
	bufferSize            int
	startedProcessingTurn bool
	firstByteReceived     bool
	generationStartTime   *time.Time
	completedSession      bool
	storedError           error
	tracingSpan           tracing.Span
}

// NewStreamedAudioResult creates a new StreamedAudioResult instance.
func NewStreamedAudioResult(
	ttsModel TTSModel,
	ttsSettings TTSModelSettings,
	voicePipelineConfig VoicePipelineConfig,
) *StreamedAudioResult {
	bufferSize := ttsSettings.BufferSize
	if bufferSize <= 0 {
		bufferSize = 120
	}

	return &StreamedAudioResult{
		ttsModel:              ttsModel,
		ttsSettings:           ttsSettings,
		totalOutputText:       "",
		instructions:          cmp.Or(ttsSettings.Instructions, DefaultTTSInstructions),
		textGenerationTask:    nil,
		voicePipelineConfig:   voicePipelineConfig,
		textBuffer:            "",
		turnTextBuffer:        "",
		queue:                 asyncqueue.New[VoiceStreamEvent](),
		tasks:                 nil,
		orderedTasks:          nil,
		dispatcherTask:        nil,
		doneProcessing:        false,
		bufferSize:            bufferSize,
		startedProcessingTurn: false,
		firstByteReceived:     false,
		generationStartTime:   nil,
		completedSession:      false,
		storedError:           nil,
		tracingSpan:           nil,
	}
}

func (r *StreamedAudioResult) startTurn(ctx context.Context) error {
	if r.startedProcessingTurn {
		return nil
	}

	r.tracingSpan = tracing.NewSpeechGroupSpan(ctx, tracing.SpeechGroupSpanParams{})
	err := r.tracingSpan.Start(ctx, false)
	if err != nil {
		return fmt.Errorf("error starting speech group span: %w", err)
	}

	r.startedProcessingTurn = true
	r.firstByteReceived = false

	r.generationStartTime = new(time.Time)
	*r.generationStartTime = time.Now()

	r.queue.Put(VoiceStreamEventLifecycle{Event: VoiceStreamEventLifecycleEventTurnStarted})

	return nil
}

func (r *StreamedAudioResult) setTask(task *asynctask.Task[any]) {
	r.textGenerationTask = task
}

func (r *StreamedAudioResult) addError(err error) {
	r.queue.Put(VoiceStreamEventError{Error: err})
}

func (r *StreamedAudioResult) transformAudioBuffer(buffer [][]byte, outputDataType AudioDataType) (AudioData, error) {
	flatLen := 0
	for _, v := range buffer {
		flatLen += len(v)
	}
	if flatLen%2 != 0 {
		return nil, fmt.Errorf("full audio buffer length %d is not even: cannot convert to []int16", flatLen)
	}

	flatBuffer := make([]byte, 0, flatLen)
	for _, v := range buffer {
		flatBuffer = append(flatBuffer, v...)
	}

	audioInt16 := make(AudioDataInt16, len(flatBuffer)/2)
	for i := range audioInt16 {
		audioInt16[i] = int16(binary.LittleEndian.Uint16(flatBuffer[i*2 : i*2+2]))
	}

	switch outputDataType {
	case AudioDataTypeInt16:
		return audioInt16, nil
	case AudioDataTypeFloat32:
		audioFloat32 := make(AudioDataFloat32, len(audioInt16))
		for i, v := range audioInt16 {
			audioFloat32[i] = float32(v) / 32767.0
		}
		return audioFloat32, nil
	default:
		return nil, UserErrorf("invalid output audio data type %v", outputDataType)
	}
}

func (r *StreamedAudioResult) streamAudio(
	ctx context.Context,
	text string,
	localQueue *asyncqueue.Queue[VoiceStreamEvent],
	finishTurn bool,
) error {
	var spanInput string
	if r.voicePipelineConfig.TraceIncludeSensitiveData.Or(true) {
		spanInput = text
	}

	return tracing.SpeechSpan(
		ctx,
		tracing.SpeechSpanParams{
			Model: r.ttsModel.ModelName(),
			Input: spanInput,
			ModelConfig: map[string]any{
				"voice":        r.ttsSettings.Voice,
				"instructions": r.instructions,
				"speed":        r.ttsSettings.Speed,
			},
			OutputFormat: "pcm",
			Parent:       r.tracingSpan,
		},
		func(ctx context.Context, ttsSpan tracing.Span) (err error) {
			defer func() {
				if err != nil {
					var errorText string
					if r.voicePipelineConfig.TraceIncludeSensitiveData.Or(true) {
						errorText = text
					}
					ttsSpan.SetError(tracing.SpanError{
						Message: err.Error(),
						Data:    map[string]any{"text": errorText},
					})
					Logger().Error("Error streaming audio", slog.String("error", err.Error()))

					// Signal completion for whole session because of error
					localQueue.Put(VoiceStreamEventLifecycle{Event: VoiceStreamEventLifecycleEventSessionEnded})
				}
			}()

			firstByteReceived := false
			var buffer [][]byte
			var fullAudioData [][]byte

			ttsModelRunResult := r.ttsModel.Run(ctx, text, r.ttsSettings)
			for chunk := range ttsModelRunResult.Seq() {
				if !firstByteReceived {
					firstByteReceived = true
					ttsSpan.SpanData().(*tracing.SpeechSpanData).FirstContentAt = time.Now().UTC().Format(time.RFC3339Nano)
				}
				if len(chunk) > 0 {
					buffer = append(buffer, chunk)
					fullAudioData = append(fullAudioData, chunk)
					if len(buffer) >= r.bufferSize {
						audioData, err := r.transformAudioBuffer(buffer, r.ttsSettings.AudioDataType.Or(AudioDataTypeInt16))
						if err != nil {
							return fmt.Errorf("error transforming audio buffer: %w", err)
						}
						if r.ttsSettings.TransformData != nil {
							audioData, err = r.ttsSettings.TransformData(ctx, audioData)
							if err != nil {
								return err
							}
						}
						localQueue.Put(VoiceStreamEventAudio{Data: audioData})
						buffer = nil
					}
				}
			}
			if err = ttsModelRunResult.Error(); err != nil {
				return fmt.Errorf("TTS model run error: %w", err)
			}

			if len(buffer) > 0 {
				audioData, err := r.transformAudioBuffer(buffer, r.ttsSettings.AudioDataType.Or(AudioDataTypeInt16))
				if err != nil {
					return fmt.Errorf("error transforming audio buffer: %w", err)
				}
				if r.ttsSettings.TransformData != nil {
					audioData, err = r.ttsSettings.TransformData(ctx, audioData)
					if err != nil {
						return err
					}
				}
				localQueue.Put(VoiceStreamEventAudio{Data: audioData})
			}

			if r.voicePipelineConfig.TraceIncludeSensitiveAudioData.Or(true) {
				ttsSpan.SpanData().(*tracing.SpeechSpanData).Output = audioToBase64(fullAudioData)
			}

			if finishTurn {
				localQueue.Put(VoiceStreamEventLifecycle{Event: VoiceStreamEventLifecycleEventTurnEnded})
			} else {
				localQueue.Put(nil) // Signal completion for this segment
			}
			return nil
		},
	)
}

func (r *StreamedAudioResult) addText(ctx context.Context, text string) error {
	err := r.startTurn(ctx)
	if err != nil {
		return err
	}

	r.textBuffer += text
	r.totalOutputText += text
	r.turnTextBuffer += text

	var combinedSentences string
	textSplitter := r.ttsSettings.TextSplitter
	if textSplitter == nil {
		textSplitter = GetTTSSentenceBasedSplitter(20)
	}

	combinedSentences, r.textBuffer, err = r.ttsSettings.TextSplitter(r.textBuffer)
	if err != nil {
		return err
	}

	if len(combinedSentences) >= 20 {
		localQueue := asyncqueue.New[VoiceStreamEvent]()
		r.orderedTasks = append(r.orderedTasks, localQueue)

		r.tasks = append(r.tasks, asynctask.CreateTask(ctx, func(ctx context.Context) (any, error) {
			return nil, r.streamAudio(ctx, combinedSentences, localQueue, false)
		}))
		if r.dispatcherTask == nil {
			r.dispatcherTask = asynctask.CreateTask(ctx, func(ctx context.Context) (any, error) {
				return nil, r.dispatchAudio(ctx)
			})
		}
	}

	return nil
}

func (r *StreamedAudioResult) turnDone(ctx context.Context) {
	if r.textBuffer != "" {
		localQueue := asyncqueue.New[VoiceStreamEvent]()
		r.orderedTasks = append(r.orderedTasks, localQueue) // Append the local queue for the final segment
		textBuffer := r.textBuffer
		r.tasks = append(r.tasks, asynctask.CreateTask(ctx, func(ctx context.Context) (any, error) {
			return nil, r.streamAudio(ctx, textBuffer, localQueue, true)
		}))
		r.textBuffer = ""
	}
	r.doneProcessing = true
	if r.dispatcherTask != nil {
		r.dispatcherTask = asynctask.CreateTask(ctx, func(ctx context.Context) (any, error) {
			return nil, r.dispatchAudio(ctx)
		})
	}

	for _, task := range r.tasks {
		task.Await()
	}
}

func (r *StreamedAudioResult) finishTurn(ctx context.Context) error {
	if r.tracingSpan != nil {
		if r.voicePipelineConfig.TraceIncludeSensitiveData.Or(true) {
			r.tracingSpan.SpanData().(*tracing.SpeechGroupSpanData).Input = r.turnTextBuffer
		}

		err := r.tracingSpan.Finish(ctx, false)
		if err != nil {
			return fmt.Errorf("error finishing stream audio tracing span: %w", err)
		}
		r.tracingSpan = nil
	}

	r.turnTextBuffer = ""
	r.startedProcessingTurn = false

	return nil
}

func (r *StreamedAudioResult) done() {
	r.completedSession = true
	r.waitForCompletion()
}

// Dispatch audio chunks from each segment in the order they were added.
func (r *StreamedAudioResult) dispatchAudio(ctx context.Context) error {
	// FIXME: data races
	for {
		if len(r.orderedTasks) == 0 {
			if r.completedSession {
				break
			}
			time.Sleep(1 * time.Nanosecond)
			continue
		}

		localQueue := r.orderedTasks[0]
		r.orderedTasks = r.orderedTasks[1:]

		for {
			chunk := localQueue.Get()
			if chunk == nil {
				break
			}
			r.queue.Put(chunk)
			if e, ok := chunk.(VoiceStreamEventLifecycle); ok && e.Event == VoiceStreamEventLifecycleEventTurnEnded {
				err := r.finishTurn(ctx)
				if err != nil {
					return err
				}
				break
			}
		}
	}

	r.queue.Put(VoiceStreamEventLifecycle{Event: VoiceStreamEventLifecycleEventSessionEnded})

	return nil
}

func (r *StreamedAudioResult) waitForCompletion() {
	for _, task := range r.tasks {
		task.Await()
	}
	if r.dispatcherTask != nil {
		r.dispatcherTask.Await()
	}
}

func (r *StreamedAudioResult) cleanupTasks(ctx context.Context) error {
	err := r.finishTurn(ctx)
	if err != nil {
		return err
	}

	for _, task := range r.tasks {
		if !task.IsDone() {
			task.Cancel()
		}
	}

	if r.dispatcherTask != nil && !r.dispatcherTask.IsDone() {
		r.dispatcherTask.Cancel()
	}

	if r.textGenerationTask != nil && !r.textGenerationTask.IsDone() {
		r.textGenerationTask.Cancel()
	}

	return nil
}

func (r *StreamedAudioResult) checkErrors() {
	for _, task := range r.tasks {
		if task.IsDone() {
			result := task.Await()
			if result.Error != nil {
				r.storedError = result.Error
				break
			}
		}
	}
}

// Stream the events and audio data as they're generated.
func (r *StreamedAudioResult) Stream(ctx context.Context) *StreamedAudioResultStream {
	return &StreamedAudioResultStream{
		ctx: ctx,
		r:   r,
	}
}

type StreamedAudioResultStream struct {
	ctx context.Context
	r   *StreamedAudioResult
	err error
}

func (s *StreamedAudioResultStream) Seq() iter.Seq[VoiceStreamEvent] {
	r := s.r
	return func(yield func(VoiceStreamEvent) bool) {
		canYield := true // once yield returns false, stop yielding, but finish consuming the events queue
		for {
			event := r.queue.Get()
			if event == nil {
				break
			}
			if e, ok := event.(VoiceStreamEventError); ok {
				r.storedError = e.Error
				Logger().Error("Error processing output", slog.String("error", e.Error.Error()))
				break
			}
			if canYield {
				canYield = yield(event)
			}
			if e, ok := event.(VoiceStreamEventLifecycle); ok && e.Event == VoiceStreamEventLifecycleEventSessionEnded {
				break
			}
		}

		r.checkErrors()
		if err := r.cleanupTasks(s.ctx); err != nil {
			r.storedError = errors.Join(r.storedError, err)
		}
	}
}

func (s *StreamedAudioResultStream) Error() error {
	return s.r.storedError
}
