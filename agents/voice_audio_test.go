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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAudioDataInt16(t *testing.T) {
	data := AudioDataInt16{-32768, -32767, -1, 0, 1, 32766, 32767}
	assert.Equal(t, 7, data.Len())
	assert.Equal(t, []byte{
		0x00, 0x80,
		0x01, 0x80,
		0xff, 0xff,
		0x00, 0x00,
		0x01, 0x00,
		0xfe, 0x7f,
		0xff, 0x7f,
	}, data.Bytes())
	assert.Equal(t, data, data.Int16())
	assert.Equal(t, []int{-32768, -32767, -1, 0, 1, 32766, 32767}, data.Int())
}

func TestAudioDataFloat32(t *testing.T) {
	data := AudioDataFloat32{-1, -0.5, 0, 0.5, 1}
	assert.Equal(t, 5, data.Len())
	assert.Equal(t, []byte{
		0x00, 0x00, 0x80, 0xbf,
		0x00, 0x00, 0x00, 0xbf,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x3f,
		0x00, 0x00, 0x80, 0x3f,
	}, data.Bytes())
	assert.Equal(t, AudioDataInt16{-32767, -16383, 0, 16383, 32767}, data.Int16())
	assert.Equal(t, []int{-32767, -16383, 0, 16383, 32767}, data.Int())
}
