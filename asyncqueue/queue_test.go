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

package asyncqueue

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQueue(t *testing.T) {
	q := New[int]()

	assert.True(t, q.IsEmpty())

	q.Put(1)
	assert.False(t, q.IsEmpty())

	q.Put(2)
	q.Put(3)

	assert.False(t, q.IsEmpty())
	assert.Equal(t, 1, q.Get())
	assert.Equal(t, 2, q.Get())
	assert.Equal(t, 3, q.Get())
	assert.True(t, q.IsEmpty())

	q.Put(4)
	q.Put(5)
	q.Put(6)

	v, ok := q.GetNoWait()
	assert.True(t, ok)
	assert.Equal(t, 4, v)

	v, ok = q.GetNoWait()
	assert.True(t, ok)
	assert.Equal(t, 5, v)

	v, ok = q.GetNoWait()
	assert.True(t, ok)
	assert.Equal(t, 6, v)

	v, ok = q.GetNoWait()
	assert.False(t, ok)
}
