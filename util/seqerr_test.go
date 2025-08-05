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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSeqErrFunc(t *testing.T) {
	customError := errors.New("error")
	s := SeqErrFunc(func(yield func(string) bool) error {
		yield("foo")
		yield("bar")
		yield("baz")
		return customError
	})

	var values []string
	for v := range s.Seq() {
		values = append(values, v)
	}
	assert.Equal(t, []string{"foo", "bar", "baz"}, values)
	assert.ErrorIs(t, s.Error(), customError)
}
