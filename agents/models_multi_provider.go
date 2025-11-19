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
	"maps"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
)

// MultiProvider is a ModelProvider that maps to a Model based on the prefix of the model name.
// By default, the mapping is:
// - "openai/" prefix or no prefix -> OpenAIProvider. e.g. "openai/gpt-4.1", "gpt-4.1"
//
//	You can override or customize this mapping.
type MultiProvider struct {
	// Optional provider map.
	ProviderMap       *MultiProviderMap
	OpenAIProvider    *OpenAIProvider
	fallbackProviders map[string]ModelProvider
}

type NewMultiProviderParams struct {
	// Optional MultiProviderMap that maps prefixes to ModelProviders. If not provided,
	// we will use a default mapping. See the documentation for MultiProvider to see the
	// default mapping.
	ProviderMap *MultiProviderMap

	// The API key to use for the OpenAI provider. If not provided, we will use
	// the default API key.
	OpenaiAPIKey param.Opt[string]

	// The base URL to use for the OpenAI provider. If not provided, we will
	// use the default base URL.
	OpenaiBaseURL param.Opt[string]

	// Optional OpenAI client to use. If not provided, we will create a new
	// OpenAI client using the OpenaiAPIKey and OpenaiBaseURL.
	OpenaiClient *OpenaiClient

	// The organization to use for the OpenAI provider.
	OpenaiOrganization param.Opt[string]

	// The project to use for the OpenAI provider.
	OpenaiProject param.Opt[string]

	// Whether to use the OpenAI responses API.
	OpenaiUseResponses param.Opt[bool]
}

// NewMultiProvider creates a new OpenAI provider.
func NewMultiProvider(params NewMultiProviderParams) *MultiProvider {
	return &MultiProvider{
		ProviderMap: params.ProviderMap,
		OpenAIProvider: NewOpenAIProvider(OpenAIProviderParams{
			APIKey:       params.OpenaiAPIKey,
			BaseURL:      params.OpenaiBaseURL,
			OpenaiClient: params.OpenaiClient,
			Organization: params.OpenaiOrganization,
			Project:      params.OpenaiProject,
			UseResponses: params.OpenaiUseResponses,
		}),
		fallbackProviders: make(map[string]ModelProvider),
	}
}

func (mp *MultiProvider) getPrefixAndModelName(modelName string) (_, _ string) {
	if modelName == "" {
		return "", ""
	}
	if prefix, name, ok := strings.Cut(modelName, "/"); ok {
		return prefix, name
	}
	return "", modelName
}

func (mp *MultiProvider) createFallbackProvider(prefix string) (ModelProvider, error) {
	// We didn't implement any fallback provider, so here we always return an error
	return nil, UserErrorf("unknown prefix %q", prefix)
}

func (mp *MultiProvider) getFallbackProvider(prefix string) (ModelProvider, error) {
	if prefix == "" || prefix == "openai" {
		return mp.OpenAIProvider, nil
	}
	if fp, ok := mp.fallbackProviders[prefix]; ok {
		return fp, nil
	}

	fp, err := mp.createFallbackProvider(prefix)
	if err != nil {
		return nil, err
	}
	mp.fallbackProviders[prefix] = fp
	return fp, nil
}

// GetModel returns a Model based on the model name. The model name can have a prefix, ending with
// a "/", which will be used to look up the ModelProvider. If there is no prefix, we will use
// the OpenAI provider.
func (mp *MultiProvider) GetModel(modelName string) (Model, error) {
	prefix, name := mp.getPrefixAndModelName(modelName)

	if prefix != "" && mp.ProviderMap != nil {
		if provider, ok := mp.ProviderMap.GetProvider(prefix); ok {
			return provider.GetModel(name)
		}
	}

	fp, err := mp.getFallbackProvider(prefix)
	if err != nil {
		return nil, err
	}
	return fp.GetModel(name)
}

// MultiProviderMap is a map of model name prefixes to ModelProvider objects.
type MultiProviderMap struct {
	m map[string]ModelProvider
}

func NewMultiProviderMap() *MultiProviderMap {
	return &MultiProviderMap{
		m: make(map[string]ModelProvider),
	}
}

// HasPrefix returns true if the given prefix is in the mapping.
func (m *MultiProviderMap) HasPrefix(prefix string) bool {
	_, ok := m.m[prefix]
	return ok
}

// GetMapping returns a copy of the current prefix -> ModelProvider mapping.
func (m *MultiProviderMap) GetMapping() map[string]ModelProvider {
	return maps.Clone(m.m)
}

// SetMapping overwrites the current mapping with a new one.
func (m *MultiProviderMap) SetMapping(mapping map[string]ModelProvider) {
	m.m = mapping
}

// GetProvider returns the ModelProvider for the given prefix.
func (m *MultiProviderMap) GetProvider(prefix string) (ModelProvider, bool) {
	v, ok := m.m[prefix]
	return v, ok
}

// AddProvider adds a new prefix -> ModelProvider mapping.
func (m *MultiProviderMap) AddProvider(prefix string, provider ModelProvider) {
	m.m[prefix] = provider
}

// RemoveProvider removes the mapping for the given prefix.
func (m *MultiProviderMap) RemoveProvider(prefix string) {
	delete(m.m, prefix)
}
