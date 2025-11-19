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
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIChatCompletionsModel_prepareRequest(t *testing.T) {
	t.Run("with ModelSettings.CustomizeChatCompletionsRequest nil", func(t *testing.T) {
		m := NewOpenAIChatCompletionsModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))

		var params *openai.ChatCompletionNewParams
		var opts []option.RequestOption

		err := tracing.GenerationSpan(
			t.Context(), tracing.GenerationSpanParams{Disabled: true},
			func(ctx context.Context, span tracing.Span) (err error) {
				params, opts, err = m.prepareRequest(
					t.Context(),
					param.Opt[string]{},
					InputString("input"),
					modelsettings.ModelSettings{
						CustomizeChatCompletionsRequest: nil,
					},
					nil,
					nil,
					nil,
					span,
					ModelTracingDisabled,
					false,
				)
				return err
			},
		)

		require.NoError(t, err)
		assert.Equal(t, &openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: param.NewOpt("input"),
					},
					Role: constant.ValueOf[constant.User](),
				},
			}},
			Model: "model-name",
		}, params)
		assert.Nil(t, opts)
	})

	t.Run("with ModelSettings.CustomizeChatCompletionsRequest returning values", func(t *testing.T) {
		customParams := &openai.ChatCompletionNewParams{
			Model: "foo",
		}
		customOpts := []option.RequestOption{
			option.WithHeader("bar", "baz"),
		}

		m := NewOpenAIChatCompletionsModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))

		var params *openai.ChatCompletionNewParams
		var opts []option.RequestOption

		err := tracing.GenerationSpan(
			t.Context(), tracing.GenerationSpanParams{Disabled: true},
			func(ctx context.Context, span tracing.Span) (err error) {
				params, opts, err = m.prepareRequest(
					t.Context(),
					param.Opt[string]{},
					InputString("input"),
					modelsettings.ModelSettings{
						CustomizeChatCompletionsRequest: func(ctx context.Context, params *openai.ChatCompletionNewParams, opts []option.RequestOption) (*openai.ChatCompletionNewParams, []option.RequestOption, error) {
							assert.Equal(t, &openai.ChatCompletionNewParams{
								Messages: []openai.ChatCompletionMessageParamUnion{{
									OfUser: &openai.ChatCompletionUserMessageParam{
										Content: openai.ChatCompletionUserMessageParamContentUnion{
											OfString: param.NewOpt("input"),
										},
										Role: constant.ValueOf[constant.User](),
									},
								}},
								Model: "model-name",
							}, params)
							assert.Nil(t, opts)
							return customParams, customOpts, nil
						},
					},
					nil,
					nil,
					nil,
					span,
					ModelTracingDisabled,
					false,
				)
				return err
			},
		)

		require.NoError(t, err)
		assert.Same(t, customParams, params)
		assert.Equal(t, customOpts, opts)
	})

	t.Run("with ModelSettings.CustomizeChatCompletionsRequest returning error", func(t *testing.T) {
		customError := errors.New("error")
		m := NewOpenAIChatCompletionsModel("model-name", NewOpenaiClient(param.Opt[string]{}, param.Opt[string]{}))

		err := tracing.GenerationSpan(
			t.Context(), tracing.GenerationSpanParams{Disabled: true},
			func(ctx context.Context, span tracing.Span) error {
				_, _, err := m.prepareRequest(
					t.Context(),
					param.Opt[string]{},
					InputString("input"),
					modelsettings.ModelSettings{
						CustomizeChatCompletionsRequest: func(ctx context.Context, params *openai.ChatCompletionNewParams, opts []option.RequestOption) (*openai.ChatCompletionNewParams, []option.RequestOption, error) {
							return nil, nil, customError
						},
					},
					nil,
					nil,
					nil,
					span,
					ModelTracingDisabled,
					false,
				)
				return err
			},
		)
		require.ErrorIs(t, err, customError)
	})
}
