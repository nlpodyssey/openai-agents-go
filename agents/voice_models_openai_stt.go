package agents

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"slices"
	"time"

	"github.com/gorilla/websocket"
	"github.com/nlpodyssey/openai-agents-go/asyncqueue"
	"github.com/nlpodyssey/openai-agents-go/asynctask"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3"
)

const (

	// VoiceModelsOpenAIEventInactivityTimeout is the timeout for inactivity in event processing.
	VoiceModelsOpenAIEventInactivityTimeout = 1000 * time.Second

	// VoiceModelsOpenAISessionCreationTimeout is the timeout waiting for session.created event
	VoiceModelsOpenAISessionCreationTimeout = 10 * time.Second

	// VoiceModelsOpenAISessionUpdateTimeout is the timeout waiting for session.updated event
	VoiceModelsOpenAISessionUpdateTimeout = 10 * time.Second
)

var voiceModelsOpenAIDefaultTurnDetection = map[string]any{"type": "semantic_vad"}

type voiceModelsOpenAIErrorSentinel struct{ err error }
type voiceModelsOpenAISessionCompleteSentinel struct{}
type voiceModelsOpenAIWebsocketDoneSentinel struct{}

func voiceModelsOpenAIAudioToBase64(audioData []AudioData) string {
	totalLen := 0
	for _, v := range audioData {
		totalLen += v.Len()
	}

	concatenatedAudio := make(AudioDataInt16, 0, totalLen)
	for _, data := range audioData {
		concatenatedAudio = append(concatenatedAudio, data.Int16()...)
	}

	audioBytes := concatenatedAudio.Bytes()

	return base64.StdEncoding.EncodeToString(audioBytes)
}

type voiceModelsOpenAITimeoutError struct{ error }

// Wait for an event from eventQueue whose type is in expectedTypes within the specified timeout.
func voiceModelsOpenAIWaitForEvent(
	eventQueue *asyncqueue.Queue[map[string]any],
	expectedTypes []string,
	timeout time.Duration,
) (map[string]any, error) {
	startTime := time.Now()
	for {
		remaining := timeout - (time.Now().Sub(startTime))
		if remaining <= 0 {
			return nil, voiceModelsOpenAITimeoutError{error: fmt.Errorf("timeout waiting for event(s): %v", expectedTypes)}
		}
		event, ok := eventQueue.GetTimeout(remaining)
		if !ok {
			continue
		}
		eventType, _ := event["type"].(string)
		if slices.Contains(expectedTypes, eventType) {
			return event, nil
		}
		if eventType == "error" {
			return nil, fmt.Errorf("error event: %v", event["error"])
		}
	}
}

// OpenAISTTTranscriptionSession is a transcription session for OpenAI's STT model.
type OpenAISTTTranscriptionSession struct {
	websocketURL                   string
	connected                      bool
	client                         OpenaiClient
	model                          string
	settings                       STTModelSettings
	turnDetection                  map[string]any
	traceIncludeSensitiveData      bool
	traceIncludeSensitiveAudioData bool

	inputQueue      *asyncqueue.Queue[AudioData]
	outputQueue     *asyncqueue.Queue[openAISTTTranscriptionSessionOutputQueueValue]
	websocket       *websocket.Conn
	eventQueue      *asyncqueue.Queue[openAISTTTranscriptionSessionEventQueueValue]
	stateQueue      *asyncqueue.Queue[map[string]any]
	turnAudioBuffer []AudioData
	tracingSpan     tracing.Span

	// tasks

	listenerTask      *asynctask.TaskNoValue
	processEventsTask *asynctask.TaskNoValue
	streamAudioTask   *asynctask.TaskNoValue
	connectionTask    *asynctask.TaskNoValue
	storedError       error
}

type openAISTTTranscriptionSessionOutputQueueValue interface {
	isOpenAISTTTranscriptionSessionOutputQueueValue()
}

type openAISTTTranscriptionSessionOutputQueueValueString string

func (openAISTTTranscriptionSessionOutputQueueValueString) isOpenAISTTTranscriptionSessionOutputQueueValue() {
}
func (voiceModelsOpenAIErrorSentinel) isOpenAISTTTranscriptionSessionOutputQueueValue()           {}
func (voiceModelsOpenAISessionCompleteSentinel) isOpenAISTTTranscriptionSessionOutputQueueValue() {}

type openAISTTTranscriptionSessionEventQueueValue interface {
	isOpenAISTTTranscriptionSessionEventQueueValue()
}

type openAISTTTranscriptionSessionEventQueueValueMap map[string]any

func (openAISTTTranscriptionSessionEventQueueValueMap) isOpenAISTTTranscriptionSessionEventQueueValue() {
}
func (voiceModelsOpenAIWebsocketDoneSentinel) isOpenAISTTTranscriptionSessionEventQueueValue() {}

type OpenAISTTTranscriptionSessionParams struct {
	Input                          StreamedAudioInput
	Client                         OpenaiClient
	Model                          string
	Settings                       STTModelSettings
	TraceIncludeSensitiveData      bool
	TraceIncludeSensitiveAudioData bool

	// Optional, defaults to DefaultOpenAISTTTranscriptionSessionWebsocketURL
	WebsocketURL string
}

const DefaultOpenAISTTTranscriptionSessionWebsocketURL = "wss://api.openai.com/v1/realtime?intent=transcription"

func NewOpenAISTTTranscriptionSession(params OpenAISTTTranscriptionSessionParams) *OpenAISTTTranscriptionSession {
	turnDetection := params.Settings.TurnDetection
	if len(turnDetection) == 0 {
		turnDetection = voiceModelsOpenAIDefaultTurnDetection
	}

	return &OpenAISTTTranscriptionSession{
		websocketURL:                   cmp.Or(params.WebsocketURL, DefaultOpenAISTTTranscriptionSessionWebsocketURL),
		connected:                      false,
		client:                         params.Client,
		model:                          params.Model,
		settings:                       params.Settings,
		turnDetection:                  turnDetection,
		traceIncludeSensitiveData:      params.TraceIncludeSensitiveData,
		traceIncludeSensitiveAudioData: params.TraceIncludeSensitiveAudioData,

		inputQueue:      params.Input.Queue,
		outputQueue:     asyncqueue.New[openAISTTTranscriptionSessionOutputQueueValue](),
		websocket:       nil,
		eventQueue:      asyncqueue.New[openAISTTTranscriptionSessionEventQueueValue](),
		stateQueue:      asyncqueue.New[map[string]any](),
		turnAudioBuffer: nil,
		tracingSpan:     nil,

		listenerTask:      nil,
		processEventsTask: nil,
		streamAudioTask:   nil,
		connectionTask:    nil,
		storedError:       nil,
	}
}

func (s *OpenAISTTTranscriptionSession) startTurn(ctx context.Context) error {
	s.tracingSpan = tracing.NewTranscriptionSpan(ctx, tracing.TranscriptionSpanParams{
		Model: s.model,
		ModelConfig: map[string]any{
			"temperature":    s.settings.Temperature,
			"language":       s.settings.Language,
			"prompt":         s.settings.Prompt,
			"turn_detection": s.turnDetection,
		},
	})
	err := s.tracingSpan.Start(ctx, false)
	if err != nil {
		return fmt.Errorf("error starting tracing span: %w", err)
	}
	return nil
}

func (s *OpenAISTTTranscriptionSession) endTurn(ctx context.Context, transcript string) error {
	if transcript == "" || s.tracingSpan == nil {
		return nil
	}

	spanData := s.tracingSpan.SpanData().(*tracing.TranscriptionSpanData)

	if s.traceIncludeSensitiveAudioData {
		spanData.Input = voiceModelsOpenAIAudioToBase64(s.turnAudioBuffer)
	}

	spanData.InputFormat = "pcm"

	if s.traceIncludeSensitiveData {
		spanData.Output = transcript
	}

	err := s.tracingSpan.Finish(ctx, false)
	if err != nil {
		return fmt.Errorf("error finishing tracing span: %w", err)
	}
	s.turnAudioBuffer = nil
	s.tracingSpan = nil
	return nil
}

func (s *OpenAISTTTranscriptionSession) eventListener(ctx context.Context) (err error) {
	if s.websocket == nil {
		return fmt.Errorf("websocket not initialized")
	}

	defer func() {
		if err != nil {
			s.outputQueue.Put(voiceModelsOpenAIErrorSentinel{err: err})
			err = STTWebsocketConnectionErrorf("error parsing events: %w", err)
		}
	}()

	for {
		_, message, err := s.websocket.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure) {
				break
			}
			return fmt.Errorf("error reading websocket message: %w", err)
		}

		var event map[string]any
		err = json.Unmarshal(message, &event)
		if err != nil {
			return fmt.Errorf("error JSON-unmarshaling websocket message: %w", err)
		}

		eventType, _ := event["type"].(string)
		if eventType == "error" {
			return STTWebsocketConnectionErrorf("error event: %v", event["error"])
		}
		if slices.Contains([]string{
			"session.updated",
			"transcription_session.updated",
			"session.created",
			"transcription_session.created",
		}, eventType) {
			s.stateQueue.Put(event)
		}

		s.eventQueue.Put(openAISTTTranscriptionSessionEventQueueValueMap(event))
	}

	s.eventQueue.Put(voiceModelsOpenAIWebsocketDoneSentinel{})
	return nil
}

func (s *OpenAISTTTranscriptionSession) configureSession() error {
	if s.websocket == nil {
		return fmt.Errorf("websocket not initialized")
	}
	return s.websocket.WriteJSON(map[string]any{
		"type": "transcription_session.update",
		"session": map[string]any{
			"input_audio_format":        "pcm16",
			"input_audio_transcription": map[string]any{"model": s.model},
			"turn_detection":            s.turnDetection,
		},
	})
}

func (s *OpenAISTTTranscriptionSession) setupConnection(ctx context.Context, c *websocket.Conn) (err error) {
	s.websocket = c
	s.listenerTask = asynctask.CreateTaskNoValue(ctx, s.eventListener)

	defer func() {
		if err != nil {
			s.outputQueue.Put(voiceModelsOpenAIErrorSentinel{err: err})
		}
	}()

	_, err = voiceModelsOpenAIWaitForEvent(
		s.stateQueue,
		[]string{"session.created", "transcription_session.created"},
		VoiceModelsOpenAISessionCreationTimeout,
	)
	if err != nil {
		if errors.As(err, &voiceModelsOpenAITimeoutError{}) {
			err = STTWebsocketConnectionErrorf("timeout waiting for transcription_session.created event: %w", err)
		}
		return err
	}

	if err = s.configureSession(); err != nil {
		return err
	}

	event, err := voiceModelsOpenAIWaitForEvent(
		s.stateQueue,
		[]string{"session.updated", "transcription_session.updated"},
		VoiceModelsOpenAISessionUpdateTimeout,
	)
	if err != nil {
		if errors.As(err, &voiceModelsOpenAITimeoutError{}) {
			err = STTWebsocketConnectionErrorf("timeout waiting for transcription_session.updated event: %w", err)
		}
		return err
	}
	if DontLogModelData {
		Logger().Debug("Session updated")
	} else {
		Logger().Debug("Session updated", slog.Any("event", event))
	}
	return nil
}

func (s *OpenAISTTTranscriptionSession) handleEvents(ctx context.Context) (err error) {
	defer func() {
		if err != nil {
			s.outputQueue.Put(voiceModelsOpenAIErrorSentinel{err: err})
		}
	}()

loop:
	for {
		event, ok := s.eventQueue.GetTimeout(VoiceModelsOpenAIEventInactivityTimeout)
		if !ok {
			// No new events for a while. Assume the session is done.
			break
		}

		switch event := event.(type) {
		case voiceModelsOpenAIWebsocketDoneSentinel:
			// processed all events and websocket is done
			break loop
		case openAISTTTranscriptionSessionEventQueueValueMap:
			eventType, _ := event["type"].(string)
			if eventType == "conversation.item.input_audio_transcription.completed" {
				transcript, _ := event["transcript"].(string)
				if transcript != "" {
					if err = s.endTurn(ctx, transcript); err != nil {
						return err
					}
					if err = s.startTurn(ctx); err != nil {
						return err
					}
					s.outputQueue.Put(openAISTTTranscriptionSessionOutputQueueValueString(transcript))
				}
			}
		default:
			// This would be an unrecoverable implementation bug, so a panic is appropriate.
			panic(fmt.Errorf("unexpected openAISTTTranscriptionSessionEventQueueValue type %T", event))
		}
	}

	s.outputQueue.Put(voiceModelsOpenAISessionCompleteSentinel{})
	return nil
}

func (s *OpenAISTTTranscriptionSession) streamAudio(ctx context.Context, audioQueue *asyncqueue.Queue[AudioData]) error {
	if s.websocket == nil {
		return fmt.Errorf("websocket not initialized")
	}
	if err := s.startTurn(ctx); err != nil {
		return err
	}

	for {
		buffer := audioQueue.Get()
		if buffer.Len() == 0 {
			break
		}

		s.turnAudioBuffer = append(s.turnAudioBuffer, buffer)

		err := s.websocket.WriteJSON(map[string]any{
			"type":  "input_audio_buffer.append",
			"audio": base64.StdEncoding.EncodeToString(buffer.Bytes()),
		})
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure) {
				break
			}
			err = fmt.Errorf("websocket wiriting error: %w", err)
			s.outputQueue.Put(voiceModelsOpenAIErrorSentinel{err: err})
			return err
		}
	}

	return nil
}

func (s *OpenAISTTTranscriptionSession) processWebsocketConnection(ctx context.Context) (err error) {
	defer func() {
		if err != nil {
			s.outputQueue.Put(voiceModelsOpenAIErrorSentinel{err: err})
		}
	}()

	header := make(http.Header)
	if s.client.APIKey.Valid() {
		header.Set("Authorization", "Bearer "+s.client.APIKey.Value)
	}
	header.Set("OpenAI-Beta", "realtime=v1")
	header.Set("OpenAI-Log-Session", "1")
	c, _, err := websocket.DefaultDialer.Dial(s.websocketURL, header)
	if err != nil {
		return fmt.Errorf("websocket connection error: %w", err)
	}
	defer func() {
		if err != nil {
			if e := c.Close(); e != nil {
				err = errors.Join(err, fmt.Errorf("error closing websocket connection: %w", e))
			}
		}
	}()

	if err = s.setupConnection(ctx, c); err != nil {
		return err
	}

	s.processEventsTask = asynctask.CreateTaskNoValue(ctx, s.handleEvents)
	s.streamAudioTask = asynctask.CreateTaskNoValue(ctx, func(ctx context.Context) error {
		return s.streamAudio(ctx, s.inputQueue)
	})
	s.connected = true

	if s.listenerTask == nil {
		Logger().Error("Listener task not initialized")
		return NewAgentsError("listener task not initialized")
	}

	s.listenerTask.Await()
	return nil
}

func (s *OpenAISTTTranscriptionSession) checkErrors() {
	tasks := []*asynctask.TaskNoValue{
		s.connectionTask,
		s.processEventsTask,
		s.streamAudioTask,
		s.listenerTask,
	}
	for _, t := range tasks {
		if t != nil && t.IsDone() {
			if err := t.Await().Error; err != nil {
				s.storedError = err
			}
		}
	}
}

func (s *OpenAISTTTranscriptionSession) cleanupTasks() {
	tasks := []*asynctask.TaskNoValue{
		s.connectionTask,
		s.processEventsTask,
		s.streamAudioTask,
		s.listenerTask,
	}
	for _, t := range tasks {
		if t != nil && !t.IsDone() {
			t.Cancel()
		}
	}
}

func (s *OpenAISTTTranscriptionSession) TranscribeTurns(ctx context.Context) StreamedTranscriptionSessionTranscribeTurns {
	return &openAISTTTranscriptionSessionTranscribeTurns{ctx: ctx, s: s}
}

func (s *OpenAISTTTranscriptionSession) Close(context.Context) (err error) {
	if s.websocket != nil {
		if err = s.websocket.Close(); err != nil {
			err = fmt.Errorf("error closing websocket connection: %w", err)
		}
	}

	s.cleanupTasks()
	return nil
}

type openAISTTTranscriptionSessionTranscribeTurns struct {
	ctx context.Context
	s   *OpenAISTTTranscriptionSession
	err error
}

func (o *openAISTTTranscriptionSessionTranscribeTurns) Seq() iter.Seq[string] {
	ctx := o.ctx
	s := o.s
	return func(yield func(string) bool) {
		canYield := true // once yield returns false, stop yielding, but finish consuming the queue

		s.connectionTask = asynctask.CreateTaskNoValue(ctx, s.processWebsocketConnection)

	loop:
		for {
			turn := s.outputQueue.Get()

			switch t := turn.(type) {
			case openAISTTTranscriptionSessionOutputQueueValueString:
				if canYield {
					canYield = yield(string(t))
				}

			case voiceModelsOpenAIErrorSentinel, voiceModelsOpenAISessionCompleteSentinel:
				break loop
			default:
				// This would be an unrecoverable implementation bug, so a panic is appropriate.
				panic(fmt.Errorf("unexpected openAISTTTranscriptionSessionOutputQueueValue type %T", t))
			}
		}

		if s.tracingSpan != nil {
			if err := s.endTurn(ctx, ""); err != nil {
				o.err = errors.Join(o.err, fmt.Errorf("error ending turn: %w", err))
				return
			}
		}

		if s.websocket != nil {
			if err := s.websocket.Close(); err != nil {
				o.err = errors.Join(o.err, fmt.Errorf("error closing websocket connection: %w", err))
			}
		}

		s.checkErrors()
		o.err = errors.Join(o.err, s.storedError)
	}
}

func (o *openAISTTTranscriptionSessionTranscribeTurns) Error() error { return o.err }

// OpenAISTTModel is a speech-to-text model for OpenAI.
type OpenAISTTModel struct {
	model  string
	client OpenaiClient
}

// NewOpenAISTTModel creates a new OpenAI speech-to-text model.
func NewOpenAISTTModel(model string, openAIClient OpenaiClient) *OpenAISTTModel {
	return &OpenAISTTModel{
		model:  model,
		client: openAIClient,
	}
}

func (m *OpenAISTTModel) ModelName() string { return m.model }

func (m *OpenAISTTModel) Transcribe(ctx context.Context, params STTModelTranscribeParams) (string, error) {
	var spanInput string
	if params.TraceIncludeSensitiveAudioData {
		spanInput = params.Input.ToBase64()
	}

	var result string
	err := tracing.TranscriptionSpan(
		ctx,
		tracing.TranscriptionSpanParams{
			Model:       m.model,
			Input:       spanInput,
			InputFormat: "pcm",
			ModelConfig: map[string]any{
				"temperature": params.Settings.Temperature,
				"language":    params.Settings.Language,
				"prompt":      params.Settings.Prompt,
			},
		},
		func(ctx context.Context, span tracing.Span) (err error) {
			spanData := span.SpanData().(*tracing.TranscriptionSpanData)

			defer func() {
				if err != nil {
					spanData.Output = ""
					span.SetError(tracing.SpanError{
						Message: err.Error(),
						Data:    nil,
					})
				}
			}()

			audioFile, err := params.Input.ToAudioFile()
			if err != nil {
				return err
			}

			response, err := m.client.Audio.Transcriptions.New(ctx, openai.AudioTranscriptionNewParams{
				Model:       m.model,
				File:        openai.File(bytes.NewReader(audioFile.Content), audioFile.Filename, audioFile.ContentType),
				Prompt:      params.Settings.Prompt,
				Language:    params.Settings.Language,
				Temperature: params.Settings.Temperature,
			})
			if err != nil {
				return fmt.Errorf("audio transcription error: %w", err)
			}

			if params.TraceIncludeSensitiveData {
				spanData.Output = response.Text
			}
			result = response.Text
			return
		},
	)
	if err != nil {
		return "", err
	}
	return result, nil
}

func (m *OpenAISTTModel) CreateSession(ctx context.Context, params STTModelCreateSessionParams) (StreamedTranscriptionSession, error) {
	return NewOpenAISTTTranscriptionSession(OpenAISTTTranscriptionSessionParams{
		Input:                          params.Input,
		Client:                         m.client,
		Model:                          m.model,
		Settings:                       params.Settings,
		TraceIncludeSensitiveData:      params.TraceIncludeSensitiveData,
		TraceIncludeSensitiveAudioData: params.TraceIncludeSensitiveAudioData,
	}), nil
}
