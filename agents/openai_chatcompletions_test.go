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
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func makeOpenaiClientWithResponse(t *testing.T, v any) OpenaiClient {
	t.Helper()

	body, err := json.Marshal(v)
	require.NoError(t, err)

	return OpenaiClient{
		BaseURL: param.NewOpt("https://fake"),
		Client: openai.NewClient(
			option.WithMiddleware(func(req *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
				return &http.Response{
					StatusCode:    http.StatusOK,
					Body:          io.NopCloser(bytes.NewReader(body)),
					ContentLength: int64(len(body)),
					Header: http.Header{
						"Content-Type": []string{"application/json"},
					},
				}, nil
			}),
		),
	}
}

func TestGetResponseWithTextMessage(t *testing.T) {
	// When the model returns a ChatCompletionMessage with plain text content,
	// `GetResponse` should produce a single `ResponseOutputMessage` containing
	// a `ResponseOutputText` with that content, and a `Usage` populated from
	// the completion's usage.

	type m = map[string]any
	msg := m{"role": "assistant", "content": "Hello"}                // ChatCompletionMessage
	choice := m{"index": 0, "finish_reason": "stop", "message": msg} // Choice
	chat := m{                                                       // ChatCompletion
		"id":      "resp-id",
		"created": 0,
		"model":   "fake",
		"object":  "chat.completion",
		"choices": []any{choice},
		"usage":   m{"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
	}
	dummyClient := makeOpenaiClientWithResponse(t, chat)

	provider := NewOpenAIProvider(OpenAIProviderParams{
		OpenaiClient: &dummyClient,
		UseResponses: param.NewOpt(false),
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	resp, err := model.GetResponse(t.Context(), ModelGetResponseParams{
		SystemInstructions: param.Null[string](),
		Input:              InputString(""),
		ModelSettings:      modelsettings.ModelSettings{},
		Tools:              nil,
		OutputSchema:       nil,
		Handoffs:           nil,
		PreviousResponseID: "",
	})
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Should have produced exactly one output message with one text part
	require.Len(t, resp.Output, 1)
	msgItem := resp.Output[0]
	assert.Equal(t, "message", msgItem.Type)
	require.Len(t, msgItem.Content, 1)
	assert.Equal(t, "output_text", msgItem.Content[0].Type)
	assert.Equal(t, "Hello", msgItem.Content[0].Text)

	// Usage should be preserved from underlying ChatCompletion.Usage
	assert.Equal(t, uint64(7), resp.Usage.InputTokens)
	assert.Equal(t, uint64(5), resp.Usage.OutputTokens)
	assert.Equal(t, uint64(12), resp.Usage.TotalTokens)
	assert.Equal(t, "", resp.ResponseID)
}

func TestGetResponseWithRefusal(t *testing.T) {
	// When the model returns a ChatCompletionMessage with a `refusal` instead
	// of normal `content`, `GetResponse` should produce a single
	// `ResponseOutputMessage` containing a `ResponseOutputRefusal` part.

	type m = map[string]any
	msg := m{"role": "assistant", "refusal": "No thanks"}            // ChatCompletionMessage
	choice := m{"index": 0, "finish_reason": "stop", "message": msg} // Choice
	chat := m{                                                       // ChatCompletion
		"id":      "resp-id",
		"created": 0,
		"model":   "fake",
		"object":  "chat.completion",
		"choices": []any{choice},
		"usage":   nil,
	}
	dummyClient := makeOpenaiClientWithResponse(t, chat)

	provider := NewOpenAIProvider(OpenAIProviderParams{
		OpenaiClient: &dummyClient,
		UseResponses: param.NewOpt(false),
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	resp, err := model.GetResponse(t.Context(), ModelGetResponseParams{
		SystemInstructions: param.Null[string](),
		Input:              InputString(""),
		ModelSettings:      modelsettings.ModelSettings{},
		Tools:              nil,
		OutputSchema:       nil,
		Handoffs:           nil,
		PreviousResponseID: "",
	})
	require.NoError(t, err)
	require.NotNil(t, resp)

	require.Len(t, resp.Output, 1)
	assert.Equal(t, "message", resp.Output[0].Type)

	require.Len(t, resp.Output[0].Content, 1)
	refusalPart := resp.Output[0].Content[0]
	assert.Equal(t, "refusal", refusalPart.Type)
	assert.Equal(t, "No thanks", refusalPart.Refusal)

	// With no usage from the completion, usage defaults to zeros.
	assert.Equal(t, uint64(0), resp.Usage.InputTokens)
	assert.Equal(t, uint64(0), resp.Usage.OutputTokens)
	assert.Equal(t, uint64(0), resp.Usage.TotalTokens)
}

func TestGetResponseWithToolCall(t *testing.T) {
	// If the ChatCompletionMessage includes one or more ToolCalls, `GetResponse`
	// should append corresponding `ResponseFunctionToolCall` items after the
	// assistant message item with matching name/arguments.

	type m = map[string]any
	toolCall := m{ // ChatCompletionMessageToolCall
		"id":       "call-id",
		"type":     "function",
		"function": m{"name": "do_thing", "arguments": `{"x":1}`}, // Function
	}
	msg := m{"role": "assistant", "content": "Hi", "tool_calls": []any{toolCall}} // ChatCompletionMessage
	choice := m{"index": 0, "finish_reason": "stop", "message": msg}              // Choice
	chat := m{                                                                    // ChatCompletion
		"id":      "resp-id",
		"created": 0,
		"model":   "fake",
		"object":  "chat.completion",
		"choices": []any{choice},
		"usage":   nil,
	}
	dummyClient := makeOpenaiClientWithResponse(t, chat)

	provider := NewOpenAIProvider(OpenAIProviderParams{
		OpenaiClient: &dummyClient,
		UseResponses: param.NewOpt(false),
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	resp, err := model.GetResponse(t.Context(), ModelGetResponseParams{
		SystemInstructions: param.Null[string](),
		Input:              InputString(""),
		ModelSettings:      modelsettings.ModelSettings{},
		Tools:              nil,
		OutputSchema:       nil,
		Handoffs:           nil,
		PreviousResponseID: "",
	})
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Expect a message item followed by a function tool call item.
	require.Len(t, resp.Output, 2)
	assert.Equal(t, "message", resp.Output[0].Type)
	fnCallItem := resp.Output[1]
	assert.Equal(t, "function_call", fnCallItem.Type)
	assert.Equal(t, "call-id", fnCallItem.CallID)
	assert.Equal(t, "do_thing", fnCallItem.Name)
	assert.Equal(t, `{"x":1}`, fnCallItem.Arguments)
}

func TestPrepareRequestNonStream(t *testing.T) {
	// Verify that `prepareRequest` builds the correct OpenAI API call when not
	// streaming.

	dummyClient := makeOpenaiClientWithResponse(t, nil)

	provider := NewOpenAIProvider(OpenAIProviderParams{
		OpenaiClient: &dummyClient,
		UseResponses: param.NewOpt(false),
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	// Execute the private prepareRequest with a system instruction and simple string input.
	params, opts, err := model.(OpenAIChatCompletionsModel).prepareRequest(
		param.NewOpt("sys"),
		InputString("hi"),
		modelsettings.ModelSettings{},
		nil,
		nil,
		nil,
		"",
		false,
	)
	require.NoError(t, err)
	assert.Nil(t, opts)
	assert.NotNil(t, params)

	assert.Equal(t, param.Null[bool](), params.Store)
	assert.Equal(t, "gpt-4", params.Model)
	assert.Equal(t, "system", *params.Messages[0].GetRole())
	assert.Equal(t, "sys", params.Messages[0].OfSystem.Content.OfString.Value)
	assert.Equal(t, "user", *params.Messages[1].GetRole())
	// Defaults for optional fields
	assert.Zero(t, params.Tools)
	assert.Zero(t, params.ToolChoice)
	assert.Zero(t, params.ResponseFormat)
	assert.Zero(t, params.StreamOptions)
}

func TestStoreParam(t *testing.T) {
	t.Run("should default to Null with no base URL", func(t *testing.T) {
		modelSettings := modelsettings.ModelSettings{}
		client := NewOpenaiClient(param.Null[string]())
		assert.Equal(t, param.Null[bool](), ChatCmplHelpers().GetStoreParam(client, modelSettings))
	})

	t.Run("for OpenAI API calls", func(t *testing.T) {
		client := NewOpenaiClient(param.NewOpt("https://api.openai.com/v1/"))

		t.Run("should default to true ", func(t *testing.T) {
			modelSettings := modelsettings.ModelSettings{}
			assert.Equal(t, param.NewOpt(true), ChatCmplHelpers().GetStoreParam(client, modelSettings))
		})

		t.Run("should respect explicitly set Store=false", func(t *testing.T) {
			modelSettings := modelsettings.ModelSettings{Store: param.NewOpt(false)}
			assert.Equal(t, param.NewOpt(false), ChatCmplHelpers().GetStoreParam(client, modelSettings))
		})

		t.Run("should respect explicitly set Store=true", func(t *testing.T) {
			modelSettings := modelsettings.ModelSettings{Store: param.NewOpt(true)}
			assert.Equal(t, param.NewOpt(true), ChatCmplHelpers().GetStoreParam(client, modelSettings))
		})
	})

	t.Run("for non-OpenAI API calls", func(t *testing.T) {
		client := NewOpenaiClient(param.NewOpt("https://example.com"))

		t.Run("should default to Null", func(t *testing.T) {
			modelSettings := modelsettings.ModelSettings{}
			assert.Equal(t, param.Null[bool](), ChatCmplHelpers().GetStoreParam(client, modelSettings))
		})

		t.Run("should respect explicitly set Store=false", func(t *testing.T) {
			modelSettings := modelsettings.ModelSettings{Store: param.NewOpt(false)}
			assert.Equal(t, param.NewOpt(false), ChatCmplHelpers().GetStoreParam(client, modelSettings))
		})

		t.Run("should respect explicitly set Store=true", func(t *testing.T) {
			modelSettings := modelsettings.ModelSettings{Store: param.NewOpt(true)}
			assert.Equal(t, param.NewOpt(true), ChatCmplHelpers().GetStoreParam(client, modelSettings))
		})
	})
}
