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
	"fmt"

	"github.com/nlpodyssey/openai-agents-go/types/optional"
)

var (
	defaultOpenaiKey      = optional.None[string]()
	defaultOpenaiClient   = optional.None[OpenaiClient]()
	useResponsesByDefault = true
)

// SetDefaultOpenaiKey sets the default OpenAI API key to use for LLM requests.
// This is only necessary if the OPENAI_API_KEY environment variable is not already set.
//
// If provided, this key will be used instead of the OPENAI_API_KEY environment variable.
func SetDefaultOpenaiKey(key string) {
	defaultOpenaiKey = optional.Value(key)
}

func GetDefaultOpenaiKey() optional.Optional[string] {
	return defaultOpenaiKey
}

// SetDefaultOpenaiClient sets the default OpenAI client to use for LLM requests.
// If provided, this client will be used instead of the default OpenAI client.
func SetDefaultOpenaiClient(client OpenaiClient) {
	defaultOpenaiClient = optional.Value(client)
}

func GetDefaultOpenaiClient() optional.Optional[OpenaiClient] {
	return defaultOpenaiClient
}

func SetUseResponsesByDefault(useResponses bool) {
	useResponsesByDefault = useResponses
}

func GetUseResponsesByDefault() bool {
	return useResponsesByDefault
}

// SetDefaultOpenaiAPI set the default API to use for OpenAI LLM requests.
// By default, we will use the responses API, but you can set this to use the
// chat completions API instead.
func SetDefaultOpenaiAPI(api OpenaiAPIType) {
	switch api {
	case OpenaiAPITypeChatCompletions:
		SetUseResponsesByDefault(false)
	case OpenaiAPITypeResponses:
		SetUseResponsesByDefault(true)
	default:
		panic(fmt.Errorf("invalid OpenaiAPIType value %q", api))
	}
}

type OpenaiAPIType string

const (
	OpenaiAPITypeChatCompletions OpenaiAPIType = "chat_completions"
	OpenaiAPITypeResponses       OpenaiAPIType = "responses"
)
