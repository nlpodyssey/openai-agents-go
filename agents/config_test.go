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
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/tracing/tracingtesting"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSetDefaultOpenaiKeyChatCompletions(t *testing.T) {
	for _, useResponses := range []bool{true, false} {
		t.Run(fmt.Sprintf("UseResponses: %v", useResponses), func(t *testing.T) {
			v := defaultOpenaiKey.Load()
			t.Cleanup(func() { defaultOpenaiKey.Store(v) })

			tracingtesting.Setup(t)
			ClearOpenaiSettings()

			SetDefaultOpenaiKey("test_key", true)

			var reqHeader http.Header
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				reqHeader = r.Header
			}))
			t.Cleanup(func() { server.Close() })

			model, err := NewOpenAIProvider(OpenAIProviderParams{
				UseResponses: param.NewOpt(useResponses),
				BaseURL:      param.NewOpt(server.URL),
			}).GetModel("gpt-4")
			require.NoError(t, err)

			_, _ = model.GetResponse(t.Context(), ModelResponseParams{Input: InputString("input")})
			assert.Equal(t, "Bearer test_key", reqHeader.Get("Authorization"))
		})
	}
}

func TestSetDefaultOpenaiClient(t *testing.T) {
	for _, useResponses := range []bool{true, false} {
		t.Run(fmt.Sprintf("UseResponses: %v", useResponses), func(t *testing.T) {
			v := defaultOpenaiClient.Load()
			t.Cleanup(func() { defaultOpenaiClient.Store(v) })

			sentinelErr := errors.New("custom client was used")

			dummyClient := OpenaiClient{
				Client: openai.NewClient(
					option.WithMiddleware(func(req *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
						return nil, sentinelErr
					}),
				),
			}

			SetDefaultOpenaiClient(dummyClient, true)

			model, err := NewOpenAIProvider(OpenAIProviderParams{
				UseResponses: param.NewOpt(useResponses),
				OpenaiClient: &dummyClient,
			}).GetModel("gpt-4")
			require.NoError(t, err)

			_, err = model.GetResponse(t.Context(), ModelResponseParams{Input: InputString("input")})
			assert.ErrorIs(t, err, sentinelErr)
		})
	}
}

func TestSetDefaultOpenaiAPI(t *testing.T) {
	v := useResponsesByDefault.Load()
	t.Cleanup(func() { useResponsesByDefault.Store(v) })

	model, err := NewOpenAIProvider(OpenAIProviderParams{}).GetModel("gpt-4")
	require.NoError(t, err)
	assert.IsType(t, OpenAIResponsesModel{}, model)

	SetDefaultOpenaiAPI(OpenaiAPITypeChatCompletions)
	model, err = NewOpenAIProvider(OpenAIProviderParams{}).GetModel("gpt-4")
	require.NoError(t, err)
	assert.IsType(t, OpenAIChatCompletionsModel{}, model)

	SetDefaultOpenaiAPI(OpenaiAPITypeResponses)
	model, err = NewOpenAIProvider(OpenAIProviderParams{}).GetModel("gpt-4")
	require.NoError(t, err)
	assert.IsType(t, OpenAIResponsesModel{}, model)
}
