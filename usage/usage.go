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

type Usage struct {
	// Total requests made to the LLM API.
	Requests uint64

	// Total input tokens sent, across all requests.
	InputTokens uint64

	// Total output tokens received, across all requests.
	OutputTokens uint64

	// Total tokens sent and received, across all requests.
	TotalTokens uint64
}

func NewUsage() *Usage {
	return new(Usage)
}

func (u *Usage) Add(other *Usage) {
	u.Requests += other.Requests
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
	u.TotalTokens += other.TotalTokens
}
