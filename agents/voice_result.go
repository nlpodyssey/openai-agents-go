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
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"slices"
	"sync"
	"sync/atomic"
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
	ttsModel           TTSModel
	ttsSettings        TTSModelSettings
	totalOutputText    *atomic.Pointer[string]
	instructions       string
	textGenerationTask *atomic.Pointer[asynctask.TaskNoValue]

	voicePipelineConfig VoicePipelineConfig
	textBuffer          *atomic.Pointer[string]
	turnTextBuffer      *atomic.Pointer[string]
	queue               *asyncqueue.Queue[VoiceStreamEvent]
	tasks               []*asynctask.TaskNoValue
	tasksMu             sync.RWMutex
	orderedTasks        []*asyncqueue.Queue[VoiceStreamEvent] // List to hold local queues for each text segment
	orderedTasksMu      sync.RWMutex
	dispatcherTask      *atomic.Pointer[asynctask.TaskNoValue] // Task to dispatch audio chunks in order

	doneProcessing        *atomic.Bool
	bufferSize            int
	startedProcessingTurn *atomic.Bool
	generationStartTime   *atomic.Pointer[time.Time]
	completedSession      *atomic.Bool
	storedError           *atomic.Pointer[error]
	tracingSpan           *atomic.Pointer[tracing.Span]
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
		ttsModel:           ttsModel,
		ttsSettings:        ttsSettings,
		totalOutputText:    newZeroValAtomicPointer[string](),
		instructions:       ttsSettings.Instructions.Or(DefaultTTSInstructions),
		textGenerationTask: new(atomic.Pointer[asynctask.TaskNoValue]),

		voicePipelineConfig: voicePipelineConfig,
		textBuffer:          newZeroValAtomicPointer[string](),
		turnTextBuffer:      newZeroValAtomicPointer[string](),
		queue:               asyncqueue.New[VoiceStreamEvent](),
		tasks:               nil,
		orderedTasks:        nil,
		dispatcherTask:      new(atomic.Pointer[asynctask.TaskNoValue]),

		doneProcessing:        new(atomic.Bool), // false
		bufferSize:            bufferSize,
		startedProcessingTurn: new(atomic.Bool),               // false
		generationStartTime:   new(atomic.Pointer[time.Time]), // nil
		completedSession:      new(atomic.Bool),               // false
		storedError:           newZeroValAtomicPointer[error](),
		tracingSpan:           newZeroValAtomicPointer[tracing.Span](),
	}
}

func (r *StreamedAudioResult) getTotalOutputText() string  { return *r.totalOutputText.Load() }
func (r *StreamedAudioResult) setTotalOutputText(v string) { r.totalOutputText.Store(&v) }
func (r *StreamedAudioResult) appendTotalOutputText(v string) {
	r.setTotalOutputText(r.getTotalOutputText() + v)
}

func (r *StreamedAudioResult) getTextGenerationTask() *asynctask.TaskNoValue {
	return r.textGenerationTask.Load()
}
func (r *StreamedAudioResult) setTextGenerationTask(v *asynctask.TaskNoValue) {
	r.textGenerationTask.Store(v)
}
func (r *StreamedAudioResult) createTextGenerationTask(ctx context.Context, fn func(context.Context) error) {
	r.setTextGenerationTask(asynctask.CreateTaskNoValue(ctx, fn))
}

func (r *StreamedAudioResult) getTextBuffer() string     { return *r.textBuffer.Load() }
func (r *StreamedAudioResult) setTextBuffer(v string)    { r.textBuffer.Store(&v) }
func (r *StreamedAudioResult) appendTextBuffer(v string) { r.setTextBuffer(r.getTextBuffer() + v) }

func (r *StreamedAudioResult) getTurnTextBuffer() string  { return *r.turnTextBuffer.Load() }
func (r *StreamedAudioResult) setTurnTextBuffer(v string) { r.turnTextBuffer.Store(&v) }
func (r *StreamedAudioResult) appendTurnTextBuffer(v string) {
	r.setTurnTextBuffer(r.getTurnTextBuffer() + v)
}

func (r *StreamedAudioResult) getDispatcherTask() *asynctask.TaskNoValue {
	return r.dispatcherTask.Load()
}
func (r *StreamedAudioResult) setDispatcherTask(v *asynctask.TaskNoValue) {
	r.dispatcherTask.Store(v)
}
func (r *StreamedAudioResult) createDispatcherTask(ctx context.Context, fn func(context.Context) error) {
	r.setDispatcherTask(asynctask.CreateTaskNoValue(ctx, fn))
}

func (r *StreamedAudioResult) appendToTasks(task *asynctask.TaskNoValue) {
	r.tasksMu.Lock()
	r.tasks = append(r.tasks, task)
	r.tasksMu.Unlock()
}
func (r *StreamedAudioResult) getTasks() []*asynctask.TaskNoValue {
	r.tasksMu.RLock()
	v := slices.Clone(r.tasks)
	r.tasksMu.RUnlock()
	return v
}

func (r *StreamedAudioResult) appendToOrderedTasks(q *asyncqueue.Queue[VoiceStreamEvent]) {
	r.orderedTasksMu.Lock()
	r.orderedTasks = append(r.orderedTasks, q)
	r.orderedTasksMu.Unlock()
}
func (r *StreamedAudioResult) popFromOrderedTasks() *asyncqueue.Queue[VoiceStreamEvent] {
	r.orderedTasksMu.Lock()
	defer r.orderedTasksMu.Unlock()

	if len(r.orderedTasks) == 0 {
		return nil
	}

	v := r.orderedTasks[0]
	r.orderedTasks = r.orderedTasks[1:]
	return v
}

func (r *StreamedAudioResult) getStoredError() error  { return *r.storedError.Load() }
func (r *StreamedAudioResult) setStoredError(v error) { r.storedError.Store(&v) }

func (r *StreamedAudioResult) getTracingSpan() tracing.Span  { return *r.tracingSpan.Load() }
func (r *StreamedAudioResult) setTracingSpan(v tracing.Span) { r.tracingSpan.Store(&v) }

func (r *StreamedAudioResult) startTurn(ctx context.Context) error {
	if r.startedProcessingTurn.Load() {
		return nil
	}

	span := tracing.NewSpeechGroupSpan(ctx, tracing.SpeechGroupSpanParams{})
	err := span.Start(ctx, false)
	if err != nil {
		return fmt.Errorf("error starting speech group span: %w", err)
	}
	r.setTracingSpan(span)

	r.startedProcessingTurn.Store(true)

	now := time.Now()
	r.generationStartTime.Store(&now)

	r.queue.Put(VoiceStreamEventLifecycle{Event: VoiceStreamEventLifecycleEventTurnStarted})

	return nil
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
			Parent:       r.getTracingSpan(),
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

	r.appendTextBuffer(text)
	r.appendTotalOutputText(text)
	r.appendTurnTextBuffer(text)

	textSplitter := r.ttsSettings.TextSplitter
	if textSplitter == nil {
		textSplitter = GetTTSSentenceBasedSplitter(20)
	}

	combinedSentences, remainingText, err := textSplitter(r.getTextBuffer())
	if err != nil {
		return err
	}

	r.setTextBuffer(remainingText)
	if len(combinedSentences) >= 20 {
		localQueue := asyncqueue.New[VoiceStreamEvent]()
		r.appendToOrderedTasks(localQueue)

		r.appendToTasks(asynctask.CreateTaskNoValue(ctx, func(ctx context.Context) error {
			return r.streamAudio(ctx, combinedSentences, localQueue, false)
		}))
		if r.getDispatcherTask() == nil {
			r.createDispatcherTask(ctx, func(ctx context.Context) error {
				return r.dispatchAudio(ctx)
			})
		}
	}

	return nil
}

func (r *StreamedAudioResult) turnDone(ctx context.Context) {
	if textBuffer := r.getTextBuffer(); textBuffer != "" {
		localQueue := asyncqueue.New[VoiceStreamEvent]()
		r.appendToOrderedTasks(localQueue) // Append the local queue for the final segment
		r.appendToTasks(asynctask.CreateTaskNoValue(ctx, func(ctx context.Context) error {
			return r.streamAudio(ctx, textBuffer, localQueue, true)
		}))
		r.setTextBuffer("")
	}
	r.doneProcessing.Store(true)
	if r.getDispatcherTask() == nil {
		r.createDispatcherTask(ctx, r.dispatchAudio)
	}

	for _, task := range r.getTasks() {
		task.Await()
	}
}

func (r *StreamedAudioResult) finishTurn(ctx context.Context) error {
	if span := r.getTracingSpan(); span != nil {
		if r.voicePipelineConfig.TraceIncludeSensitiveData.Or(true) {
			span.SpanData().(*tracing.SpeechGroupSpanData).Input = r.getTurnTextBuffer()
		}

		err := span.Finish(ctx, false)
		if err != nil {
			return fmt.Errorf("error finishing stream audio tracing span: %w", err)
		}
		r.setTracingSpan(nil)
	}

	r.setTurnTextBuffer("")
	r.startedProcessingTurn.Store(false)

	return nil
}

func (r *StreamedAudioResult) done() {
	r.completedSession.Store(true)
	r.waitForCompletion()
}

// Dispatch audio chunks from each segment in the order they were added.
func (r *StreamedAudioResult) dispatchAudio(ctx context.Context) error {
	for {
		localQueue := r.popFromOrderedTasks()
		if localQueue == nil {
			if r.completedSession.Load() {
				break
			}
			time.Sleep(1 * time.Nanosecond)
			continue
		}

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
	for _, task := range r.getTasks() {
		task.Await()
	}
	if t := r.getDispatcherTask(); t != nil {
		t.Await()
	}
}

func (r *StreamedAudioResult) cleanupTasks(ctx context.Context) error {
	err := r.finishTurn(ctx)
	if err != nil {
		return err
	}

	for _, task := range r.getTasks() {
		if !task.IsDone() {
			task.Cancel()
		}
	}

	if t := r.getDispatcherTask(); t != nil && !t.IsDone() {
		t.Cancel()
	}
	if t := r.getTextGenerationTask(); t != nil && !t.IsDone() {
		t.Cancel()
	}

	return nil
}

func (r *StreamedAudioResult) checkErrors() {
	for _, task := range r.getTasks() {
		if task.IsDone() {
			result := task.Await()
			if result.Error != nil {
				r.setStoredError(result.Error)
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
				r.setStoredError(e.Error)
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
			r.setStoredError(errors.Join(r.getStoredError(), err))
		}
	}
}

func (s *StreamedAudioResultStream) Error() error {
	return s.r.getStoredError()
}
