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

package util

import (
	"errors"
	"io"
	"slices"
)

// WriteSeekerBuffer is a buffer of bytes which satisfies io.WriteSeeker interface.
type WriteSeekerBuffer struct {
	b []byte
	i int64
}

func (b *WriteSeekerBuffer) Bytes() []byte { return b.b }

func (b *WriteSeekerBuffer) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	end := b.i + int64(len(p))
	if n := end - int64(cap(b.b)); n > 0 {
		b.b = slices.Grow(b.b, int(n))
	}
	if end > int64(len(b.b)) {
		b.b = b.b[:end]
	}
	copy(b.b[b.i:end], p)
	b.i = end
	return len(p), nil
}

func (b *WriteSeekerBuffer) Seek(offset int64, whence int) (int64, error) {
	var newOffset int64
	switch whence {
	case io.SeekStart:
		newOffset = offset
	case io.SeekCurrent:
		newOffset = b.i + offset
	case io.SeekEnd:
		newOffset = int64(len(b.b)) + offset
	default:
		return 0, errors.New("WriteSeekerBuffer.Seek: invalid whence")
	}
	if newOffset < 0 {
		return 0, errors.New("WriteSeekerBuffer.Seek: negative position")
	}
	b.i = newOffset
	return newOffset, nil
}
