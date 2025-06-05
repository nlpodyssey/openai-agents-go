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

package usage

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUsage_Add(t *testing.T) {
	u := &Usage{
		Requests:     1,
		InputTokens:  2,
		OutputTokens: 3,
		TotalTokens:  4,
	}
	other := &Usage{
		Requests:     50,
		InputTokens:  60,
		OutputTokens: 70,
		TotalTokens:  80,
	}
	u.Add(other)

	expected := &Usage{
		Requests:     51,
		InputTokens:  62,
		OutputTokens: 73,
		TotalTokens:  84,
	}
	assert.Equal(t, expected, u)
}
