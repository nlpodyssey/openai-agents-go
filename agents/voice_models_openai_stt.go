package agents

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"iter"
	"slices"
	"time"

	"github.com/nlpodyssey/openai-agents-go/asyncqueue"
	"github.com/nlpodyssey/openai-agents-go/asynctask"
	"github.com/nlpodyssey/openai-agents-go/tracing"
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
	flatLen := 0
	for _, v := range audioData {
		flatLen += v.Len()
	}

	concatenatedAudio := make([]int16, 0, flatLen)
	for _, data := range audioData {
		switch d := data.(type) {
		case AudioDataInt16:
			concatenatedAudio = append(concatenatedAudio, d...)
		case AudioDataFloat32:
			for _, v := range d {
				concatenatedAudio = append(concatenatedAudio, int16(min(1, max(-1, v))*32767))
			}
		default:
			// This would be an unrecoverable implementation bug, so a panic is appropriate.
			panic(fmt.Errorf("unexpected AudioData type %T", d))
		}
	}

	audioBytes := make([]byte, len(concatenatedAudio)*2)
	for i, v := range concatenatedAudio {
		binary.LittleEndian.PutUint16(audioBytes[i*2:], uint16(v))
	}

	return base64.StdEncoding.EncodeToString(audioBytes)
}

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
			return nil, fmt.Errorf("timeout waiting for event(s): %v", expectedTypes)
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
	client                         OpenaiClient
	model                          string
	settings                       STTModelSettings
	turnDetection                  map[string]any
	traceIncludeSensitiveData      bool
	traceIncludeSensitiveAudioData bool

	inputQueue  *asyncqueue.Queue[AudioData]
	outputQueue *asyncqueue.Queue[openAISTTTranscriptionSessionOutputQueueValue]
	// TODO: self._websocket: websockets.ClientConnection | None = None
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
}

func NewOpenAISTTTranscriptionSession(params OpenAISTTTranscriptionSessionParams) *OpenAISTTTranscriptionSession {
	turnDetection := params.Settings.TurnDetection
	if len(turnDetection) == 0 {
		turnDetection = voiceModelsOpenAIDefaultTurnDetection
	}

	return &OpenAISTTTranscriptionSession{
		client:                         params.Client,
		model:                          params.Model,
		settings:                       params.Settings,
		turnDetection:                  turnDetection,
		traceIncludeSensitiveData:      params.TraceIncludeSensitiveData,
		traceIncludeSensitiveAudioData: params.TraceIncludeSensitiveAudioData,

		inputQueue:      params.Input.Queue,
		outputQueue:     asyncqueue.New[openAISTTTranscriptionSessionOutputQueueValue](),
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

	spanData.InputFormat = "PCM"

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

//    async def _event_listener(self) -> None:
//        assert self._websocket is not None, "Websocket not initialized"
//
//        async for message in self._websocket:
//            try:
//                event = json.loads(message)
//
//                if event.get("type") == "error":
//                    raise STTWebsocketConnectionError(f"Error event: {event.get('error')}")
//
//                if event.get("type") in [
//                    "session.updated",
//                    "transcription_session.updated",
//                    "session.created",
//                    "transcription_session.created",
//                ]:
//                    await self._state_queue.put(event)
//
//                await self._event_queue.put(event)
//            except Exception as e:
//                await self._output_queue.put(ErrorSentinel(e))
//                raise STTWebsocketConnectionError("Error parsing events") from e
//        await self._event_queue.put(WebsocketDoneSentinel())

//    async def _configure_session(self) -> None:
//        assert self._websocket is not None, "Websocket not initialized"
//        await self._websocket.send(
//            json.dumps(
//                {
//                    "type": "transcription_session.update",
//                    "session": {
//                        "input_audio_format": "pcm16",
//                        "input_audio_transcription": {"model": self._model},
//                        "turn_detection": self._turn_detection,
//                    },
//                }
//            )
//        )

//    async def _setup_connection(self, ws: websockets.ClientConnection) -> None:
//        self._websocket = ws
//        self._listener_task = asyncio.create_task(self._event_listener())
//
//        try:
//            event = await _wait_for_event(
//                self._state_queue,
//                ["session.created", "transcription_session.created"],
//                SESSION_CREATION_TIMEOUT,
//            )
//        except TimeoutError as e:
//            wrapped_err = STTWebsocketConnectionError(
//                "Timeout waiting for transcription_session.created event"
//            )
//            await self._output_queue.put(ErrorSentinel(wrapped_err))
//            raise wrapped_err from e
//        except Exception as e:
//            await self._output_queue.put(ErrorSentinel(e))
//            raise e
//
//        await self._configure_session()
//
//        try:
//            event = await _wait_for_event(
//                self._state_queue,
//                ["session.updated", "transcription_session.updated"],
//                SESSION_UPDATE_TIMEOUT,
//            )
//            if _debug.DONT_LOG_MODEL_DATA:
//                logger.debug("Session updated")
//            else:
//                logger.debug(f"Session updated: {event}")
//        except TimeoutError as e:
//            wrapped_err = STTWebsocketConnectionError(
//                "Timeout waiting for transcription_session.updated event"
//            )
//            await self._output_queue.put(ErrorSentinel(wrapped_err))
//            raise wrapped_err from e
//        except Exception as e:
//            await self._output_queue.put(ErrorSentinel(e))
//            raise

//    async def _handle_events(self) -> None:
//        while True:
//            try:
//                event = await asyncio.wait_for(
//                    self._event_queue.get(), timeout=EVENT_INACTIVITY_TIMEOUT
//                )
//                if isinstance(event, WebsocketDoneSentinel):
//                    # processed all events and websocket is done
//                    break
//
//                event_type = event.get("type", "unknown")
//                if event_type == "input_audio_transcription_completed":
//                    transcript = cast(str, event.get("transcript", ""))
//                    if len(transcript) > 0:
//                        self._end_turn(transcript)
//                        self._start_turn()
//                        await self._output_queue.put(transcript)
//                await asyncio.sleep(0)  # yield control
//            except asyncio.TimeoutError:
//                # No new events for a while. Assume the session is done.
//                break
//            except Exception as e:
//                await self._output_queue.put(ErrorSentinel(e))
//                raise e
//        await self._output_queue.put(SessionCompleteSentinel())

//    async def _stream_audio(
//        self, audio_queue: asyncio.Queue[npt.NDArray[np.int16 | np.float32]]
//    ) -> None:
//        assert self._websocket is not None, "Websocket not initialized"
//        self._start_turn()
//        while True:
//            buffer = await audio_queue.get()
//            if buffer is None:
//                break
//
//            self._turn_audio_buffer.append(buffer)
//            try:
//                await self._websocket.send(
//                    json.dumps(
//                        {
//                            "type": "input_audio_buffer.append",
//                            "audio": base64.b64encode(buffer.tobytes()).decode("utf-8"),
//                        }
//                    )
//                )
//            except websockets.ConnectionClosed:
//                break
//            except Exception as e:
//                await self._output_queue.put(ErrorSentinel(e))
//                raise e
//
//            await asyncio.sleep(0)  # yield control

func (s *OpenAISTTTranscriptionSession) processWebsocketConnection(ctx context.Context) (err error) {
	defer func() {
		if err != nil {
			s.outputQueue.Put(voiceModelsOpenAIErrorSentinel{err: err})
		}
	}()

	// TODO:
	//  async with websockets.connect(
	//      "wss://api.openai.com/v1/realtime?intent=transcription",
	//      additional_headers={
	//          "Authorization": f"Bearer {self._client.api_key}",
	//          "OpenAI-Beta": "realtime=v1",
	//          "OpenAI-Log-Session": "1",
	//      },
	//  ) as ws:
	//      await self._setup_connection(ws)
	//      self._process_events_task = asyncio.create_task(self._handle_events())
	//      self._stream_audio_task = asyncio.create_task(self._stream_audio(self._input_queue))
	//      self.connected = True
	//      if self._listener_task:
	//          await self._listener_task
	//      else:
	//          logger.error("Listener task not initialized")
	//          raise AgentsException("Listener task not initialized")
	panic("implement me") //TODO implement me
}

//    def _check_errors(self) -> None:
//        if self._connection_task and self._connection_task.done():
//            exc = self._connection_task.exception()
//            if exc and isinstance(exc, Exception):
//                self._stored_exception = exc
//
//        if self._process_events_task and self._process_events_task.done():
//            exc = self._process_events_task.exception()
//            if exc and isinstance(exc, Exception):
//                self._stored_exception = exc
//
//        if self._stream_audio_task and self._stream_audio_task.done():
//            exc = self._stream_audio_task.exception()
//            if exc and isinstance(exc, Exception):
//                self._stored_exception = exc
//
//        if self._listener_task and self._listener_task.done():
//            exc = self._listener_task.exception()
//            if exc and isinstance(exc, Exception):
//                self._stored_exception = exc

//    def _cleanup_tasks(self) -> None:
//        if self._listener_task and not self._listener_task.done():
//            self._listener_task.cancel()
//
//        if self._process_events_task and not self._process_events_task.done():
//            self._process_events_task.cancel()
//
//        if self._stream_audio_task and not self._stream_audio_task.done():
//            self._stream_audio_task.cancel()
//
//        if self._connection_task and not self._connection_task.done():
//            self._connection_task.cancel()

func (s *OpenAISTTTranscriptionSession) TranscribeTurns(ctx context.Context) StreamedTranscriptionSessionTranscribeTurns {
	return &openAISTTTranscriptionSessionTranscribeTurns{ctx: ctx, s: s}
}

func (s *OpenAISTTTranscriptionSession) Close(ctx context.Context) error {
	//async def close(self) -> None:
	//    if self._websocket:
	//        await self._websocket.close()
	//
	//    self._cleanup_tasks()
	panic("implement me") //TODO implement me
}

type openAISTTTranscriptionSessionTranscribeTurns struct {
	ctx context.Context
	s   *OpenAISTTTranscriptionSession
}

func (o *openAISTTTranscriptionSessionTranscribeTurns) Seq() iter.Seq[string] {
	ctx := o.ctx
	s := o.s
	return func(yield func(string) bool) {
		s.connectionTask = asynctask.CreateTaskNoValue(ctx, s.processWebsocketConnection)

		//while True:
		//    try:
		//        turn = await self._output_queue.get()
		//    except asyncio.CancelledError:
		//        break
		//
		//    if (
		//        turn is None
		//        or isinstance(turn, ErrorSentinel)
		//        or isinstance(turn, SessionCompleteSentinel)
		//    ):
		//        self._output_queue.task_done()
		//        break
		//    yield turn
		//    self._output_queue.task_done()
		//
		//if self._tracing_span:
		//    self._end_turn("")
		//
		//if self._websocket:
		//    await self._websocket.close()
		//
		//self._check_errors()
		//if self._stored_exception:
		//    raise self._stored_exception
		panic("implement me") //TODO implement me
	}
}

func (o *openAISTTTranscriptionSessionTranscribeTurns) Error() error {
	panic("implement me") //TODO implement me
}

//class OpenAISTTModel(STTModel):
//    """A speech-to-text model for OpenAI."""
//
//    def __init__(
//        self,
//        model: str,
//        openai_client: AsyncOpenAI,
//    ):
//        """Create a new OpenAI speech-to-text model.
//
//        Args:
//            model: The name of the model to use.
//            openai_client: The OpenAI client to use.
//        """
//        self.model = model
//        self._client = openai_client
//
//    @property
//    def model_name(self) -> str:
//        return self.model
//
//    def _non_null_or_not_given(self, value: Any) -> Any:
//        return value if value is not None else None  # NOT_GIVEN
//
//    async def transcribe(
//        self,
//        input: AudioInput,
//        settings: STTModelSettings,
//        trace_include_sensitive_data: bool,
//        trace_include_sensitive_audio_data: bool,
//    ) -> str:
//        """Transcribe an audio input.
//
//        Args:
//            input: The audio input to transcribe.
//            settings: The settings to use for the transcription.
//
//        Returns:
//            The transcribed text.
//        """
//        with transcription_span(
//            model=self.model,
//            input=input.to_base64() if trace_include_sensitive_audio_data else "",
//            input_format="pcm",
//            model_config={
//                "temperature": self._non_null_or_not_given(settings.temperature),
//                "language": self._non_null_or_not_given(settings.language),
//                "prompt": self._non_null_or_not_given(settings.prompt),
//            },
//        ) as span:
//            try:
//                response = await self._client.audio.transcriptions.create(
//                    model=self.model,
//                    file=input.to_audio_file(),
//                    prompt=self._non_null_or_not_given(settings.prompt),
//                    language=self._non_null_or_not_given(settings.language),
//                    temperature=self._non_null_or_not_given(settings.temperature),
//                )
//                if trace_include_sensitive_data:
//                    span.span_data.output = response.text
//                return response.text
//            except Exception as e:
//                span.span_data.output = ""
//                span.set_error(SpanError(message=str(e), data={}))
//                raise e
//
//    async def create_session(
//        self,
//        input: StreamedAudioInput,
//        settings: STTModelSettings,
//        trace_include_sensitive_data: bool,
//        trace_include_sensitive_audio_data: bool,
//    ) -> StreamedTranscriptionSession:
//        """Create a new transcription session.
//
//        Args:
//            input: The audio input to transcribe.
//            settings: The settings to use for the transcription.
//            trace_include_sensitive_data: Whether to include sensitive data in traces.
//            trace_include_sensitive_audio_data: Whether to include sensitive audio data in traces.
//
//        Returns:
//            A new transcription session.
//        """
//        return OpenAISTTTranscriptionSession(
//            input,
//            self._client,
//            self.model,
//            settings,
//            trace_include_sensitive_data,
//            trace_include_sensitive_audio_data,
//        )
