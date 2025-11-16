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
	"cmp"
	"errors"
	"os"

	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
)

const (
	DefaultSTTModel = "gpt-4o-transcribe"
	DefaultTTSModel = "gpt-4o-mini-tts"
)

// OpenAIVoiceModelProvider is a voice model provider that uses OpenAI models.
type OpenAIVoiceModelProvider struct {
	params OpenAIVoiceModelProviderParams
	client *OpenaiClient
}

type OpenAIVoiceModelProviderParams struct {
	// The API key to use for the OpenAI client. If not provided, we will use the
	// default API key.
	APIKey param.Opt[string]

	// The base URL to use for the OpenAI client. If not provided, we will use the
	// default base URL.
	BaseURL param.Opt[string]

	// An optional OpenAI client to use. If not provided, we will create a new
	// OpenAI client using the APIKey and BaseURL.
	OpenaiClient *OpenaiClient

	// The organization to use for the OpenAI client.
	Organization param.Opt[string]

	// The project to use for the OpenAI client.
	Project param.Opt[string]
}

func NewDefaultOpenAIVoiceModelProvider() *OpenAIVoiceModelProvider {
	return NewOpenAIVoiceModelProvider(OpenAIVoiceModelProviderParams{})
}

// NewOpenAIVoiceModelProvider creates a new OpenAI voice model provider.
func NewOpenAIVoiceModelProvider(params OpenAIVoiceModelProviderParams) *OpenAIVoiceModelProvider {
	if params.OpenaiClient != nil && (params.APIKey.Valid() || params.BaseURL.Valid()) {
		panic(errors.New("OpenAIVoiceModelProvider: don't provide APIKey or BaseURL if you provide OpenaiClient"))
	}
	return &OpenAIVoiceModelProvider{
		params: params,
		client: params.OpenaiClient,
	}
}

// We lazy load the client in case you never actually use OpenAIVoiceModelProvider.
// It panics if you don't have an API key set.
func (provider *OpenAIVoiceModelProvider) getClient() OpenaiClient {
	if provider.client == nil {
		if defaultClient := GetDefaultOpenaiClient(); defaultClient != nil {
			provider.client = defaultClient
		} else {
			var apiKey param.Opt[string]
			if provider.params.APIKey.Valid() {
				apiKey = provider.params.APIKey
			} else if defaultKey := GetDefaultOpenaiKey(); defaultKey.Valid() {
				apiKey = defaultKey
			} else if envKey := os.Getenv("OPENAI_API_KEY"); envKey != "" {
				apiKey = param.NewOpt(envKey)
			} else {
				Logger().Warn("OpenAIVoiceModelProvider: an API key is missing")
			}

			options := make([]option.RequestOption, 0)

			if provider.params.Organization.Valid() {
				options = append(options, option.WithOrganization(provider.params.Organization.Value))
			}
			if provider.params.Project.Valid() {
				options = append(options, option.WithProject(provider.params.Project.Value))
			}

			newClient := NewOpenaiClient(provider.params.BaseURL, apiKey, options...)
			provider.client = &newClient
		}
	}
	return *provider.client
}

func (provider *OpenAIVoiceModelProvider) GetSTTModel(modelName string) (STTModel, error) {
	return NewOpenAISTTModel(cmp.Or(modelName, DefaultSTTModel), provider.getClient()), nil
}

func (provider *OpenAIVoiceModelProvider) GetTTSModel(modelName string) (TTSModel, error) {
	return NewOpenAITTSModel(cmp.Or(modelName, DefaultTTSModel), provider.getClient()), nil
}
