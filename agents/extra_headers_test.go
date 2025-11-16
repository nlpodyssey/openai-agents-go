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

package agents_test

import (
	"bytes"
	"io"
	"net/http"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/assert"
)

func TestExtraHeadersPassedToOpenaiResponsesModel(t *testing.T) {
	// Ensure ExtraHeaders in ModelSettings is passed to the OpenAIResponsesModel client.

	var reqHeader http.Header
	dummyClient := agents.OpenaiClient{
		Client: openai.NewClient(
			option.WithMiddleware(func(req *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
				reqHeader = req.Header
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(bytes.NewReader(nil)),
				}, nil
			}),
		),
	}

	extraHeaders := map[string]string{"X-Test-Header": "test-value"}

	model := agents.NewOpenAIResponsesModel("gpt-4", dummyClient)
	_, _ = model.GetResponse(t.Context(), agents.ModelResponseParams{
		Input: agents.InputString("hi"),
		ModelSettings: modelsettings.ModelSettings{
			ExtraHeaders: extraHeaders,
		},
		Tracing: agents.ModelTracingDisabled,
	})

	assert.Equal(t, "test-value", reqHeader.Get("X-Test-Header"))
}

func TestExtraHeadersPassedToOpenaiChatCompletionsClient(t *testing.T) {
	// Ensure ExtraHeaders in ModelSettings is passed to the OpenAI chat completions client.

	var reqHeader http.Header
	dummyClient := agents.OpenaiClient{
		Client: openai.NewClient(
			option.WithMiddleware(func(req *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
				reqHeader = req.Header
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(bytes.NewReader(nil)),
				}, nil
			}),
		),
	}

	extraHeaders := map[string]string{"X-Test-Header": "test-value"}

	model := agents.NewOpenAIChatCompletionsModel("gpt-4", dummyClient)
	_, _ = model.GetResponse(t.Context(), agents.ModelResponseParams{
		Input: agents.InputString("hi"),
		ModelSettings: modelsettings.ModelSettings{
			ExtraHeaders: extraHeaders,
		},
		Tracing: agents.ModelTracingDisabled,
	})

	assert.Equal(t, "test-value", reqHeader.Get("X-Test-Header"))
}
