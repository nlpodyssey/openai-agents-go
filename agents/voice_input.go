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

import "github.com/nlpodyssey/openai-agents-go/asyncqueue"

const (
	DefaultAudioSampleRate  = 24000
	DefaultAudioSampleWidth = 2
	DefaultAudioChannels    = 1
)

// def _buffer_to_audio_file(
//    buffer: npt.NDArray[np.int16 | np.float32],
//    frame_rate: int = DEFAULT_SAMPLE_RATE,
//    sample_width: int = 2,
//    channels: int = 1,
//) -> tuple[str, io.BytesIO, str]:
//    if buffer.dtype == np.float32:
//        # convert to int16
//        buffer = np.clip(buffer, -1.0, 1.0)
//        buffer = (buffer * 32767).astype(np.int16)
//    elif buffer.dtype != np.int16:
//        raise UserError("Buffer must be a numpy array of int16 or float32")
//
//    audio_file = io.BytesIO()
//    with wave.open(audio_file, "w") as wav_file:
//        wav_file.setnchannels(channels)
//        wav_file.setsampwidth(sample_width)
//        wav_file.setframerate(frame_rate)
//        wav_file.writeframes(buffer.tobytes())
//        audio_file.seek(0)
//
//    # (filename, bytes, content_type)
//    return ("audio.wav", audio_file, "audio/wav")

// AudioInput represents static audio to be used as input for the VoicePipeline.
type AudioInput struct {
	// A buffer containing the audio data for the agent.
	Buffer AudioData

	// The sample rate of the audio data.Defaults to DefaultAudioSampleRate.
	FrameRate int

	// The sample width of the audio data. Defaults to DefaultAudioSampleWidth.
	SampleWidth int

	// The number of channels in the audio data. Defaults to DefaultAudioChannels.
	Channels int

	//def to_audio_file(self) -> tuple[str, io.BytesIO, str]:
	//    """Returns a tuple of (filename, bytes, content_type)"""
	//    return _buffer_to_audio_file(self.buffer, self.frame_rate, self.sample_width, self.channels)

	//def to_base64(self) -> str:
	//    """Returns the audio data as a base64 encoded string."""
	//    if self.buffer.dtype == np.float32:
	//        # convert to int16
	//        self.buffer = np.clip(self.buffer, -1.0, 1.0)
	//        self.buffer = (self.buffer * 32767).astype(np.int16)
	//    elif self.buffer.dtype != np.int16:
	//        raise UserError("Buffer must be a numpy array of int16 or float32")
	//
	//    return base64.b64encode(self.buffer.tobytes()).decode("utf-8")
}

// StreamedAudioInput is an audio input represented as a stream of audio data.
// You can pass this to the VoicePipeline and then push audio data into the
// queue using the AddAudio method.
type StreamedAudioInput struct {
	queue *asyncqueue.Queue[AudioData]
}

func NewStreamedAudioInput() StreamedAudioInput {
	return StreamedAudioInput{
		queue: asyncqueue.New[AudioData](),
	}
}

// AddAudio adds more audio data to the stream.
func (s StreamedAudioInput) AddAudio(audio AudioData) {
	s.queue.Put(audio)
}
