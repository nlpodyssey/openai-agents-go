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
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/go-audio/wav"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sineWaveInt16 creates a simple sine wave audio data in int16 format
func sineWaveInt16(freq float64) AudioDataInt16 {
	bufferInt64 := make(AudioDataInt16, DefaultAudioSampleRate)
	for i := range DefaultAudioSampleRate {
		lin := float64(i) / (DefaultAudioSampleRate - 1)
		bufferInt64[i] = int16(math.Sin(2*math.Pi*freq*lin) * 32767)
	}
	return bufferInt64
}

// sineWaveFloat32 creates a simple sine wave audio data in int16 format
func sineWaveFloat32(freq float64) AudioDataFloat32 {
	bufferFloat32 := make(AudioDataFloat32, DefaultAudioSampleRate)
	for i := range DefaultAudioSampleRate {
		lin := float64(i) / (DefaultAudioSampleRate - 1)
		bufferFloat32[i] = float32(math.Sin(2 * math.Pi * freq * lin))
	}
	return bufferFloat32
}

func Test_bufferToAudioFile(t *testing.T) {
	for _, buffer := range []AudioData{sineWaveInt16(440), sineWaveFloat32(440)} {
		t.Run(fmt.Sprintf("%T", buffer), func(t *testing.T) {
			audioFile, err := bufferToAudioFile(buffer, 0, 0, 0)
			require.NoError(t, err)

			assert.Equal(t, "audio.wav", audioFile.Filename)
			assert.Equal(t, "audio/wav", audioFile.ContentType)
			assert.NotEmpty(t, audioFile.Content)

			// Verify the WAV file contents
			dec := wav.NewDecoder(bytes.NewReader(audioFile.Content))
			dec.ReadInfo()
			require.NoError(t, dec.Err())

			assert.Equal(t, uint16(1), dec.NumChans)
			assert.Equal(t, uint16(16), dec.BitDepth)
			assert.Equal(t, uint32(DefaultAudioSampleRate), dec.SampleRate)

			intBuf, err := dec.FullPCMBuffer()
			require.NoError(t, err)
			assert.Equal(t, 16, intBuf.SourceBitDepth)
			assert.Equal(t, DefaultAudioSampleRate, intBuf.Format.SampleRate)
			assert.Equal(t, 1, intBuf.Format.NumChannels)
			assert.Equal(t, buffer.Len(), intBuf.NumFrames())
		})
	}
}

func TestAudioInput_ToAudioFile(t *testing.T) {
	buffer := sineWaveFloat32(440)

	audioInput := AudioInput{Buffer: buffer}
	audioFile, err := audioInput.ToAudioFile()
	require.NoError(t, err)

	assert.Equal(t, "audio.wav", audioFile.Filename)
	assert.Equal(t, "audio/wav", audioFile.ContentType)
	assert.NotEmpty(t, audioFile.Content)

	// Verify the WAV file contents
	dec := wav.NewDecoder(bytes.NewReader(audioFile.Content))
	dec.ReadInfo()
	require.NoError(t, dec.Err())

	assert.Equal(t, uint16(1), dec.NumChans)
	assert.Equal(t, uint16(16), dec.BitDepth)
	assert.Equal(t, uint32(DefaultAudioSampleRate), dec.SampleRate)

	intBuf, err := dec.FullPCMBuffer()
	require.NoError(t, err)
	assert.Equal(t, 16, intBuf.SourceBitDepth)
	assert.Equal(t, DefaultAudioSampleRate, intBuf.Format.SampleRate)
	assert.Equal(t, 1, intBuf.Format.NumChannels)
	assert.Equal(t, buffer.Len(), intBuf.NumFrames())
}

func TestNewStreamedAudioInput(t *testing.T) {
	streamedInput := NewStreamedAudioInput()

	// Create some test audio data
	audio1 := sineWaveFloat32(440)
	audio2 := sineWaveFloat32(880)

	// Add audio to the queue
	streamedInput.AddAudio(audio1)
	streamedInput.AddAudio(audio2)

	v, ok := streamedInput.Queue.GetNoWait()
	require.True(t, ok)
	assert.Equal(t, audio1, v)

	v, ok = streamedInput.Queue.GetNoWait()
	require.True(t, ok)
	assert.Equal(t, audio2, v)

	assert.True(t, streamedInput.Queue.IsEmpty())
}
