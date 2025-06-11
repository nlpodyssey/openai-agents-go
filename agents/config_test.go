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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
