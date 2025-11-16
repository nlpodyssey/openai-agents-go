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
	"sync"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
)

func TestUsage_Add(t *testing.T) {
	u := &Usage{
		Requests:    1,
		InputTokens: 2,
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: 3,
		},
		OutputTokens: 4,
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: 5,
		},
		TotalTokens: 6,
	}
	other := &Usage{
		Requests:    40,
		InputTokens: 50,
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: 60,
		},
		OutputTokens: 70,
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: 80,
		},
		TotalTokens: 90,
	}
	u.Add(other)

	expected := &Usage{
		Requests:    41,
		InputTokens: 52,
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: 63,
		},
		OutputTokens: 74,
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: 85,
		},
		TotalTokens: 96,
	}
	assert.Equal(t, expected, u)
}

func TestUsage_AddConcurrent(t *testing.T) {
	const goroutines = 128

	u := NewUsage()
	other := &Usage{
		Requests:    1,
		InputTokens: 2,
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: 3,
		},
		OutputTokens: 4,
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: 5,
		},
		TotalTokens: 6,
	}

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			u.Add(other)
		}()
	}

	wg.Wait()

	expected := &Usage{
		Requests:    uint64(goroutines),
		InputTokens: uint64(2 * goroutines),
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: int64(3 * goroutines),
		},
		OutputTokens: uint64(4 * goroutines),
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: int64(5 * goroutines),
		},
		TotalTokens: uint64(6 * goroutines),
	}

	assert.Equal(t, expected, u)
}
