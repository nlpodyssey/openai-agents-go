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
	"context"
	"errors"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIResponsesModel_prepareRequest(t *testing.T) {
	t.Run("with ModelSettings.CustomizeResponsesRequest nil", func(t *testing.T) {
		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		params, opts, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("input"),
			modelsettings.ModelSettings{
				CustomizeResponsesRequest: nil,
			},
			nil,
			nil,
			nil,
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.NoError(t, err)
		assert.Equal(t, &responses.ResponseNewParams{
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: responses.ResponseInputParam{{
					OfMessage: &responses.EasyInputMessageParam{
						Content: responses.EasyInputMessageContentUnionParam{
							OfString: param.NewOpt("input"),
						},
						Role: responses.EasyInputMessageRoleUser,
						Type: responses.EasyInputMessageTypeMessage,
					},
				}},
			},
			Model: "model-name",
		}, params)
		assert.Nil(t, opts)
	})

	t.Run("with ModelSettings.CustomizeResponsesRequest returning values", func(t *testing.T) {
		customParams := &responses.ResponseNewParams{
			Model: "foo",
		}
		customOpts := []option.RequestOption{
			option.WithHeader("bar", "baz"),
		}

		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		params, opts, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("input"),
			modelsettings.ModelSettings{
				CustomizeResponsesRequest: func(ctx context.Context, params *responses.ResponseNewParams, opts []option.RequestOption) (*responses.ResponseNewParams, []option.RequestOption, error) {
					assert.Equal(t, &responses.ResponseNewParams{
						Input: responses.ResponseNewParamsInputUnion{
							OfInputItemList: responses.ResponseInputParam{{
								OfMessage: &responses.EasyInputMessageParam{
									Content: responses.EasyInputMessageContentUnionParam{
										OfString: param.NewOpt("input"),
									},
									Role: responses.EasyInputMessageRoleUser,
									Type: responses.EasyInputMessageTypeMessage,
								},
							}},
						},
						Model: "model-name",
					}, params)
					assert.Nil(t, opts)
					return customParams, customOpts, nil
				},
			},
			nil,
			nil,
			nil,
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.NoError(t, err)
		assert.Same(t, customParams, params)
		assert.Equal(t, customOpts, opts)
	})

	t.Run("with ModelSettings.CustomizeResponsesRequest returning error", func(t *testing.T) {
		customError := errors.New("error")
		m := NewOpenAIResponsesModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))
		_, _, err := m.prepareRequest(
			t.Context(),
			param.Opt[string]{},
			InputString("input"),
			modelsettings.ModelSettings{
				CustomizeResponsesRequest: func(ctx context.Context, params *responses.ResponseNewParams, opts []option.RequestOption) (*responses.ResponseNewParams, []option.RequestOption, error) {
					return nil, nil, customError
				},
			},
			nil,
			nil,
			nil,
			"",
			false,
			responses.ResponsePromptParam{},
		)
		require.ErrorIs(t, err, customError)
	})
}
