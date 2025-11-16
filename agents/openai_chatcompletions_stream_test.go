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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func makeOpenaiClientWithStreamResponse(t *testing.T, chunks ...any) agents.OpenaiClient {
	t.Helper()

	var body bytes.Buffer
	for _, chunk := range chunks {
		jsonChunk, err := json.Marshal(chunk)
		require.NoError(t, err)
		body.WriteString("data: ")
		body.Write(jsonChunk)
		body.WriteString("\n\n")
	}
	body.WriteString("data: [DONE]\n\n")

	return agents.OpenaiClient{
		BaseURL: param.NewOpt("https://fake"),
		Client: openai.NewClient(
			option.WithMiddleware(func(req *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
				return &http.Response{
					StatusCode:    http.StatusOK,
					Body:          io.NopCloser(&body),
					ContentLength: int64(body.Len()),
					Header: http.Header{
						"Content-Type": []string{"application/json"},
					},
				}, nil
			}),
		),
	}
}

func TestStreamResponseYieldsEventsForTextContent(t *testing.T) {
	// Validate that `StreamResponse` emits the correct sequence of events when
	// streaming a simple assistant message consisting of plain text content.
	// We simulate two chunks of text returned from the chat completion stream.

	// Create two chunks that will be emitted by the fake stream.
	type m = map[string]any
	chunk1 := m{ // ChatCompletionChunk
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"content": "He"}}}, // Choice / ChoiceDelta
	}
	// Mark last chunk with usage so stream_response knows this is final.
	chunk2 := m{ // ChatCompletionChunk
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"content": "llo"}}}, // Choice / ChoiceDelta
		"usage": m{ // CompletionUsage
			"completion_tokens":         5,
			"prompt_tokens":             7,
			"total_tokens":              12,
			"prompt_tokens_details":     m{"cached_tokens": 2},    // PromptTokensDetails
			"completion_tokens_details": m{"reasoning_tokens": 3}, // CompletionTokensDetails
		},
	}

	dummyClient := makeOpenaiClientWithStreamResponse(t, chunk1, chunk2)

	provider := agents.NewOpenAIProvider(agents.OpenAIProviderParams{
		OpenaiClient: &dummyClient,
		UseResponses: param.NewOpt(false),
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	var outputEvents []agents.TResponseStreamEvent
	err = model.StreamResponse(
		t.Context(),
		agents.ModelResponseParams{
			SystemInstructions: param.Opt[string]{},
			Input:              agents.InputString(""),
			ModelSettings:      modelsettings.ModelSettings{},
			Tools:              nil,
			OutputType:         nil,
			Handoffs:           nil,
			Tracing:            agents.ModelTracingDisabled,
			PreviousResponseID: "",
			Prompt:             responses.ResponsePromptParam{},
		},
		func(ctx context.Context, event agents.TResponseStreamEvent) error {
			outputEvents = append(outputEvents, event)
			return nil
		},
	)
	require.NoError(t, err)

	// We expect a response.created, then a response.output_item.added, content part added,
	// two content delta events (for "He" and "llo"), a content part done, the assistant message
	// output_item.done, and finally response.completed.
	// There should be 8 events in total.
	require.Len(t, outputEvents, 8)

	// First event indicates creation.
	assert.Equal(t, "response.created", outputEvents[0].Type)

	// The output item added and content part added events should mark the assistant message.
	assert.Equal(t, "response.output_item.added", outputEvents[1].Type)
	assert.Equal(t, "response.content_part.added", outputEvents[2].Type)

	// Two text delta events.
	assert.Equal(t, "response.output_text.delta", outputEvents[3].Type)
	assert.Equal(t, "He", outputEvents[3].Delta)
	assert.Equal(t, "response.output_text.delta", outputEvents[4].Type)
	assert.Equal(t, "llo", outputEvents[4].Delta)

	// After streaming, the content part and item should be marked done.
	assert.Equal(t, "response.content_part.done", outputEvents[5].Type)
	assert.Equal(t, "response.output_item.done", outputEvents[6].Type)

	// Last event indicates completion of the stream.
	assert.Equal(t, "response.completed", outputEvents[7].Type)

	// The completed response should have one output message with full text.
	completedResp := outputEvents[7].Response
	assert.Equal(t, "message", completedResp.Output[0].Type)
	assert.Equal(t, "output_text", completedResp.Output[0].Content[0].Type)
	assert.Equal(t, "Hello", completedResp.Output[0].Content[0].Text)

	assert.Equal(t, int64(7), completedResp.Usage.InputTokens)
	assert.Equal(t, int64(5), completedResp.Usage.OutputTokens)
	assert.Equal(t, int64(12), completedResp.Usage.TotalTokens)
	assert.Equal(t, int64(2), completedResp.Usage.InputTokensDetails.CachedTokens)
	assert.Equal(t, int64(3), completedResp.Usage.OutputTokensDetails.ReasoningTokens)
}

func TestStreamResponseYieldsEventsForRefusalContent(t *testing.T) {
	// Validate that when the model streams a refusal string instead of normal content,
	// `StreamResponse` emits the appropriate sequence of events including
	// `response.refusal.delta` events for each chunk of the refusal message and
	// constructs a completed assistant message with a `ResponseOutputRefusal` part.

	// Simulate refusal text coming in two pieces, like content but using the `refusal`
	// field on the delta rather than `content`.
	type m = map[string]any
	chunk1 := m{ // ChatCompletionChunk
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"refusal": "No"}}}, // Choice / ChoiceDelta
	}
	chunk2 := m{ // ChatCompletionChunk
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"refusal": "Thanks"}}},               // Choice / ChoiceDelta
		"usage":   m{"completion_tokens": 2, "prompt_tokens": 2, "total_tokens": 4}, // CompletionUsage
	}

	dummyClient := makeOpenaiClientWithStreamResponse(t, chunk1, chunk2)

	provider := agents.NewOpenAIProvider(agents.OpenAIProviderParams{
		OpenaiClient: &dummyClient,
		UseResponses: param.NewOpt(false),
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	var outputEvents []agents.TResponseStreamEvent
	err = model.StreamResponse(
		t.Context(),
		agents.ModelResponseParams{
			SystemInstructions: param.Opt[string]{},
			Input:              agents.InputString(""),
			ModelSettings:      modelsettings.ModelSettings{},
			Tools:              nil,
			OutputType:         nil,
			Handoffs:           nil,
			Tracing:            agents.ModelTracingDisabled,
			PreviousResponseID: "",
			Prompt:             responses.ResponsePromptParam{},
		},
		func(ctx context.Context, event agents.TResponseStreamEvent) error {
			outputEvents = append(outputEvents, event)
			return nil
		},
	)
	require.NoError(t, err)

	// Expect sequence similar to text: created, output_item.added, content part added,
	// two refusal delta events, content part done, output_item.done, completed.
	require.Len(t, outputEvents, 8)
	assert.Equal(t, "response.created", outputEvents[0].Type)
	assert.Equal(t, "response.output_item.added", outputEvents[1].Type)
	assert.Equal(t, "response.content_part.added", outputEvents[2].Type)
	assert.Equal(t, "response.refusal.delta", outputEvents[3].Type)
	assert.Equal(t, "No", outputEvents[3].Delta)
	assert.Equal(t, "response.refusal.delta", outputEvents[4].Type)
	assert.Equal(t, "Thanks", outputEvents[4].Delta)
	assert.Equal(t, "response.content_part.done", outputEvents[5].Type)
	assert.Equal(t, "response.output_item.done", outputEvents[6].Type)
	assert.Equal(t, "response.completed", outputEvents[7].Type)

	completedResp := outputEvents[7].Response
	assert.Equal(t, "message", completedResp.Output[0].Type)
	assert.Equal(t, "refusal", completedResp.Output[0].Content[0].Type)
	assert.Equal(t, "NoThanks", completedResp.Output[0].Content[0].Refusal)
}

func TestStreamResponseYieldsEventsForToolCall(t *testing.T) {
	// Validate that `StreamResponse` emits the correct sequence of events when
	// the model is streaming a function/tool call instead of plain text.
	// The function call will be split across two chunks.

	// Simulate a single tool call whose ID stays constant and function name/args built over chunks.
	type m = map[string]any
	toolCallDelta1 := m{ // ChoiceDeltaToolCall
		"index":    0,
		"id":       "tool-id",
		"function": m{"name": "my_", "arguments": "arg1"}, // ChoiceDeltaToolCallFunction
		"type":     "function",
	}
	toolCallDelta2 := m{ // ChoiceDeltaToolCall
		"index":    0,
		"id":       "tool-id",
		"function": m{"name": "func", "arguments": "arg2"}, // ChoiceDeltaToolCallFunction
		"type":     "function",
	}
	chunk1 := m{ // ChatCompletionChunk
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"tool_calls": []m{toolCallDelta1}}}}, // Choice / ChoiceDelta
	}
	chunk2 := m{ // ChatCompletionChunk
		"id":      "chunk-id",
		"created": 1,
		"model":   "fake",
		"object":  "chat.completion.chunk",
		"choices": []m{{"index": 0, "delta": m{"tool_calls": []m{toolCallDelta2}}}}, // Choice / ChoiceDelta
		"usage":   m{"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}, // CompletionUsage
	}

	dummyClient := makeOpenaiClientWithStreamResponse(t, chunk1, chunk2)

	provider := agents.NewOpenAIProvider(agents.OpenAIProviderParams{
		OpenaiClient: &dummyClient,
		UseResponses: param.NewOpt(false),
	})
	model, err := provider.GetModel("gpt-4")
	require.NoError(t, err)

	var outputEvents []agents.TResponseStreamEvent
	err = model.StreamResponse(
		t.Context(),
		agents.ModelResponseParams{
			SystemInstructions: param.Opt[string]{},
			Input:              agents.InputString(""),
			ModelSettings:      modelsettings.ModelSettings{},
			Tools:              nil,
			Handoffs:           nil,
			Tracing:            agents.ModelTracingDisabled,
			PreviousResponseID: "",
			Prompt:             responses.ResponsePromptParam{},
		},
		func(ctx context.Context, event agents.TResponseStreamEvent) error {
			outputEvents = append(outputEvents, event)
			return nil
		},
	)
	require.NoError(t, err)

	// Sequence should be: response.created, then after loop we expect function call-related events:
	// one response.output_item.added for function call, a response.function_call_arguments.delta,
	// a response.output_item.done, and finally response.completed.
	require.Len(t, outputEvents, 5)
	assert.Equal(t, "response.created", outputEvents[0].Type)

	// The next three events are about the tool call.
	assert.Equal(t, "response.output_item.added", outputEvents[1].Type)
	// The added item should be a ResponseFunctionToolCall.
	addedFn := outputEvents[1].Item
	assert.Equal(t, "function_call", addedFn.Type)
	assert.Equal(t, "my_func", addedFn.Name) // Name should be concatenation of both chunks.
	assert.Equal(t, "arg1arg2", addedFn.Arguments)
	assert.Equal(t, "response.function_call_arguments.delta", outputEvents[2].Type)
	assert.Equal(t, "arg1arg2", outputEvents[2].Delta)
	assert.Equal(t, "response.output_item.done", outputEvents[3].Type)
	assert.Equal(t, "response.completed", outputEvents[4].Type)
}
