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
	"log/slog"
	"os"

	"github.com/nlpodyssey/openai-agents-go/types/optional"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
)

type OpenAIProviderParams struct {
	// The API key to use for the OpenAI client. If not provided, we will use the
	// default API key.
	APIKey param.Opt[string]

	// The base URL to use for the OpenAI client. If not provided, we will use the
	// default base URL.
	BaseURL optional.Optional[string]

	// An optional OpenAI client to use. If not provided, we will create a new
	// OpenAI client using the APIKey and BaseURL.
	OpenaiClient optional.Optional[OpenaiClient]

	// The organization to use for the OpenAI client.
	Organization optional.Optional[string]

	// The project to use for the OpenAI client.
	Project optional.Optional[string]

	// Whether to use the OpenAI responses API.
	UseResponses optional.Optional[bool]
}

type OpenAIProvider struct {
	params       OpenAIProviderParams
	useResponses bool
	client       optional.Optional[OpenaiClient]
}

// NewOpenAIProvider creates a new OpenAI provider.
func NewOpenAIProvider(params OpenAIProviderParams) *OpenAIProvider {
	if params.OpenaiClient.Present && (params.APIKey.Valid() || params.BaseURL.Present) {
		panic(errors.New("OpenAIProvider: don't provide APIKey or BaseURL if you provide OpenaiClient"))
	}

	return &OpenAIProvider{
		params: params,
		useResponses: params.UseResponses.ValueOrFallbackFunc(func() bool {
			return GetUseResponsesByDefault()
		}),
		client: params.OpenaiClient,
	}
}

func (provider *OpenAIProvider) GetModel(modelName string) (Model, error) {
	if modelName == "" {
		return nil, fmt.Errorf("cannot get OpenAI model without a name")
	}

	client := provider.getClient()

	if provider.useResponses {
		return NewOpenAIResponsesModel(modelName, client), nil
	}
	return NewOpenAIChatCompletionsModel(modelName, client), nil
}

// We lazy load the client in case you never actually use OpenAIProvider.
// It panics if you don't have an API key set.
func (provider *OpenAIProvider) getClient() OpenaiClient {
	if !provider.client.Present {
		provider.client = optional.Value(
			GetDefaultOpenaiClient().ValueOrFallbackFunc(func() OpenaiClient {
				var apiKey string
				if provider.params.APIKey.Valid() {
					apiKey = provider.params.APIKey.Value
				} else {
					apiKey = GetDefaultOpenaiKey().ValueOrFallbackFunc(func() string {
						v := os.Getenv("OPENAI_API_KEY")
						if v == "" {
							slog.Warn("OpenAIProvider: an API key is missing")
						}
						return v
					})
				}

				options := make([]option.RequestOption, 0)
				options = append(options, option.WithAPIKey(apiKey))

				if v, ok := provider.params.Organization.Get(); ok {
					options = append(options, option.WithOrganization(v))
				}
				if v, ok := provider.params.Project.Get(); ok {
					options = append(options, option.WithProject(v))
				}

				return NewOpenaiClient(provider.params.BaseURL, options...)
			}),
		)
	}
	return provider.client.Value
}
