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
	"encoding/binary"
	"math"
)

type AudioDataType byte

const (
	AudioDataTypeInt16 = iota + 1
	AudioDataTypeFloat32
)

type AudioData interface {
	Len() int
	Bytes() []byte
	Int16() AudioDataInt16
	Int() []int
}

type AudioDataInt16 []int16

func (d AudioDataInt16) Len() int { return len(d) }

func (d AudioDataInt16) Bytes() []byte {
	b := make([]byte, len(d)*2)
	for i, v := range d {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(v))
	}
	return b
}

func (d AudioDataInt16) Int16() AudioDataInt16 { return d }

func (d AudioDataInt16) Int() []int {
	result := make([]int, len(d))
	for i, v := range d {
		result[i] = int(v)
	}
	return result
}

type AudioDataFloat32 []float32

func (d AudioDataFloat32) Len() int { return len(d) }

func (d AudioDataFloat32) Bytes() []byte {
	b := make([]byte, 4*len(d))
	for i, v := range d {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func (d AudioDataFloat32) Int16() AudioDataInt16 {
	result := make(AudioDataInt16, len(d))
	for i, v := range d {
		result[i] = int16(min(1, max(-1, v)) * 32767)
	}
	return result
}

func (d AudioDataFloat32) Int() []int {
	result := make([]int, len(d))
	for i, v := range d {
		result[i] = int(min(1, max(-1, v)) * 32767)
	}
	return result
}
