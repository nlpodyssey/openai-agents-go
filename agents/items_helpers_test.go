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
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/openaitypes"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// makeMessage is a helper to construct a TResponseOutputItem (responses.ResponseOutputMessage)
// with a single batch of content items, using a fixed ID/Status.
func makeMessage(contentItems ...responses.ResponseOutputMessageContentUnion) agents.TResponseOutputItem {
	return agents.TResponseOutputItem{ // responses.ResponseOutputMessage
		ID:      "msg123",
		Content: contentItems,
		Role:    constant.ValueOf[constant.Assistant](),
		Status:  "completed",
		Type:    "message",
	}
}

func TestExtractLastContentOfTextMessage(t *testing.T) {
	// Build a message containing two text segments.
	message := makeMessage(
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
			Annotations: nil,
			Text:        "Hello ",
			Type:        "output_text",
		},
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
			Annotations: nil,
			Text:        "world!",
			Type:        "output_text",
		},
	)

	// Helpers should yield the last segment's text.
	v, err := agents.ItemHelpers().ExtractLastContent(message)
	require.NoError(t, err)
	assert.Equal(t, "world!", v)
}

func TestExtractLastContentOfRefusalMessage(t *testing.T) {
	// Build a message whose last content entry is a refusal.
	message := makeMessage(
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
			Annotations: nil,
			Text:        "Before refusal",
			Type:        "output_text",
		},
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputRefusal
			Refusal: "I cannot do that",
			Type:    "refusal",
		},
	)

	// Helpers should extract the refusal string when last content is a refusal.
	v, err := agents.ItemHelpers().ExtractLastContent(message)
	require.NoError(t, err)
	assert.Equal(t, "I cannot do that", v)
}

func TestExtractLastContentNonMessageReturnsEmpty(t *testing.T) {
	// Construct some other type of output item, e.g. a tool call, to verify non-message returns "".
	toolCall := agents.TResponseOutputItem{ // responses.ResponseOutputMessage
		ID:        "tool123",
		Arguments: "{}",
		CallID:    "call123",
		Name:      "func",
		Type:      "function_call",
	}

	v, err := agents.ItemHelpers().ExtractLastContent(toolCall)
	require.NoError(t, err)
	assert.Equal(t, "", v)
}

func TestExtractLastTextReturnsTextOnly(t *testing.T) {
	firstText := responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
		Annotations: nil,
		Text:        "part1",
		Type:        "output_text",
	}

	// A message whose last segment is text yields the text.
	message := makeMessage(
		firstText,
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
			Annotations: nil,
			Text:        "part2",
			Type:        "output_text",
		},
	)

	v, ok := agents.ItemHelpers().ExtractLastText(message)
	assert.True(t, ok)
	assert.Equal(t, "part2", v)

	// Whereas when last content is a refusal, ExtractLastText returns "" and false.
	message = makeMessage(
		firstText,
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputRefusal
			Refusal: "no",
			Type:    "refusal",
		},
	)

	v, ok = agents.ItemHelpers().ExtractLastText(message)
	assert.False(t, ok)
	assert.Equal(t, "", v)
}

func TestInputToNewInputListFromString(t *testing.T) {
	result := agents.ItemHelpers().InputToNewInputList(agents.InputString("hi"))

	// Should wrap the string into a list with a single item containing content and user role.
	assert.Equal(t, []agents.TResponseInputItem{
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("hi"),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
	}, result)
}

func TestInputToNewInputListCopiesLists(t *testing.T) {
	// Given a list of message items, ensure the returned list is a copy.
	original := []agents.TResponseInputItem{
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt("abc"),
				},
				Role: responses.EasyInputMessageRoleDeveloper,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
	}
	newList := agents.ItemHelpers().InputToNewInputList(agents.InputItems(original))
	assert.Equal(t, original, newList)

	// Mutating the returned list should not mutate the original.
	newList[0] = agents.TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt("def"),
			},
			Role: responses.EasyInputMessageRoleSystem,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
	assert.Equal(t, "abc", original[0].OfMessage.Content.OfString.Value)
}

func TestTextMessageOutputConcatenatesTextSegments(t *testing.T) {
	// Build a message with both text and refusal segments, only text segments are concatenated.
	message := makeMessage(
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
			Annotations: nil,
			Text:        "a",
			Type:        "output_text",
		},
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputRefusal
			Refusal: "denied",
			Type:    "refusal",
		},
		responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
			Annotations: nil,
			Text:        "b",
			Type:        "output_text",
		},
	)
	// Wrap into MessageOutputItem to feed into TextMessageOutput.
	item := agents.MessageOutputItem{
		Agent:   &agents.Agent{Name: "test"},
		RawItem: openaitypes.ResponseOutputMessageFromResponseOutputItemUnion(message),
		Type:    "message_output_item",
	}

	v := agents.ItemHelpers().TextMessageOutput(item)
	assert.Equal(t, "ab", v)
}

func TestTextMessageOutputsAcrossListOfRunItems(t *testing.T) {
	// Compose several RunItem instances, including a non-message run item, and ensure
	// that only MessageOutputItem instances contribute any text. The non-message
	// (ReasoningItem) should be ignored by Helpers.TextMessageOutputs.

	message1 := makeMessage(responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
		Annotations: nil,
		Text:        "foo",
		Type:        "output_text",
	})
	message2 := makeMessage(responses.ResponseOutputMessageContentUnion{ // responses.ResponseOutputText
		Annotations: nil,
		Text:        "bar",
		Type:        "output_text",
	})

	var item1 agents.RunItem = agents.MessageOutputItem{
		Agent:   &agents.Agent{Name: "test"},
		RawItem: openaitypes.ResponseOutputMessageFromResponseOutputItemUnion(message1),
		Type:    "message_output_item",
	}
	var item2 agents.RunItem = agents.MessageOutputItem{
		Agent:   &agents.Agent{Name: "test"},
		RawItem: openaitypes.ResponseOutputMessageFromResponseOutputItemUnion(message2),
		Type:    "message_output_item",
	}

	// Create a non-message run item of a different type, e.g., a reasoning trace.
	reasoning := responses.ResponseReasoningItem{
		ID:      "rid",
		Summary: nil,
		Type:    constant.ValueOf[constant.Reasoning](),
	}
	var nonMessageItem agents.RunItem = agents.ReasoningItem{
		Agent:   &agents.Agent{Name: "test"},
		RawItem: reasoning,
		Type:    "reasoning_item",
	}

	// Confirm only the message outputs are concatenated.
	v := agents.ItemHelpers().TextMessageOutputs([]agents.RunItem{item1, nonMessageItem, item2})
	assert.Equal(t, "foobar", v)
}

func TestToolCallOutputItemConstructsFunctionCallOutput(t *testing.T) {
	call := agents.ResponseFunctionToolCall{
		ID:        "call-abc",
		Arguments: `{"x": 1}`,
		CallID:    "call-abc",
		Name:      "do_something",
		Type:      constant.ValueOf[constant.FunctionCall](),
	}

	payload := agents.ItemHelpers().ToolCallOutputItem(call, "result-string")
	assert.Equal(t, constant.ValueOf[constant.FunctionCallOutput](), payload.Type)
	assert.Equal(t, "call-abc", payload.CallID)
	require.True(t, payload.Output.OfString.Valid())
	assert.Equal(t, "result-string", payload.Output.OfString.Value)
}

/*
The following tests ensure that every possible output item type defined by OpenAI's API
can be converted back into an input item via ModelResponse.ToInputItems.
*/

func TestToInputItemsForMessage(t *testing.T) {
	// An output message should convert into an input type matching the message's own structure.

	content := responses.ResponseOutputMessageContentUnion{
		Annotations: nil,
		Text:        "hello world",
		Type:        "output_text",
	}
	message := agents.TResponseOutputItem{ // responses.ResponseOutputMessage
		ID:      "m1",
		Content: []responses.ResponseOutputMessageContentUnion{content},
		Role:    constant.ValueOf[constant.Assistant](),
		Status:  "completed",
		Type:    "message",
	}
	resp := agents.ModelResponse{
		Output:     []agents.TResponseOutputItem{message},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	inputItems := resp.ToInputItems()

	// The value should contain exactly the primitive values of the message
	assert.Equal(t, []agents.TResponseInputItem{
		{
			OfOutputMessage: &responses.ResponseOutputMessageParam{
				ID: "m1",
				Content: []responses.ResponseOutputMessageContentUnionParam{{
					OfOutputText: &responses.ResponseOutputTextParam{
						Annotations: nil,
						Text:        "hello world",
						Type:        constant.ValueOf[constant.OutputText](),
					},
				}},
				Status: responses.ResponseOutputMessageStatusCompleted,
				Role:   constant.ValueOf[constant.Assistant](),
				Type:   constant.ValueOf[constant.Message](),
			},
		},
	}, inputItems)
}

func TestToInputItemsForFunctionCall(t *testing.T) {
	// A function tool call output should produce the same value as a function tool call input.

	toolCall := responses.ResponseOutputItemUnion{ // responses.ResponseFunctionToolCall
		ID:        "f1",
		Arguments: "{}",
		CallID:    "c1",
		Name:      "func",
		Type:      "function_call",
	}
	resp := agents.ModelResponse{
		Output:     []agents.TResponseOutputItem{toolCall},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	inputItems := resp.ToInputItems()

	// The value should contain exactly the primitive values of the message
	assert.Equal(t, []agents.TResponseInputItem{
		{
			OfFunctionCall: &responses.ResponseFunctionToolCallParam{
				ID:        param.NewOpt("f1"),
				Arguments: "{}",
				CallID:    "c1",
				Name:      "func",
				Type:      constant.ValueOf[constant.FunctionCall](),
				Status:    "",
			},
		},
	}, inputItems)
}

func TestToInputItemsForFileSearchCall(t *testing.T) {
	// A file search tool call output should produce the same value as a file search input.

	fsCall := responses.ResponseOutputItemUnion{ // responses.ResponseFileSearchToolCall
		ID:      "fs1",
		Queries: []string{"query"},
		Status:  "completed",
		Type:    "file_search_call",
	}
	resp := agents.ModelResponse{
		Output:     []agents.TResponseOutputItem{fsCall},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	inputItems := resp.ToInputItems()

	// The value should contain exactly the primitive values of the message
	assert.Equal(t, []agents.TResponseInputItem{
		{
			OfFileSearchCall: &responses.ResponseFileSearchToolCallParam{
				ID:      "fs1",
				Queries: []string{"query"},
				Status:  responses.ResponseFileSearchToolCallStatusCompleted,
				Type:    constant.ValueOf[constant.FileSearchCall](),
			},
		},
	}, inputItems)
}

func TestToInputItemsForWebSearchCall(t *testing.T) {
	// A web search tool call output should produce the same value as a web search input.
	wsCall := agents.TResponseOutputItem{ // responses.ResponseFunctionWebSearch
		ID: "w1",
		Action: responses.ResponseOutputItemUnionAction{
			Type:  "search",
			Query: "query",
		},
		Status: "completed",
		Type:   "web_search_call",
	}
	resp := agents.ModelResponse{
		Output:     []agents.TResponseOutputItem{wsCall},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}
	inputItems := resp.ToInputItems()
	assert.Equal(t, []agents.TResponseInputItem{
		{
			OfWebSearchCall: &responses.ResponseFunctionWebSearchParam{
				ID: "w1",
				Action: responses.ResponseFunctionWebSearchActionUnionParam{
					OfSearch: &responses.ResponseFunctionWebSearchActionSearchParam{
						Query: "query",
						Type:  constant.ValueOf[constant.Search](),
					},
				},
				Status: responses.ResponseFunctionWebSearchStatusCompleted,
				Type:   constant.ValueOf[constant.WebSearchCall](),
			},
		},
	}, inputItems)
}

func TestToInputItemsForComputerCall(t *testing.T) {
	action := responses.ResponseOutputItemUnionAction{ // responses.ResponseComputerToolCallActionUnion
		Type: "screenshot",
	}
	compCall := responses.ResponseOutputItemUnion{ // responses.ResponseComputerToolCall
		ID:                  "comp1",
		Action:              action,
		Type:                "computer_call",
		CallID:              "comp1",
		PendingSafetyChecks: nil,
		Status:              "completed",
	}
	resp := agents.ModelResponse{
		Output:     []agents.TResponseOutputItem{compCall},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	inputItems := resp.ToInputItems()

	// The value should contain exactly the primitive values of the message
	assert.Equal(t, []agents.TResponseInputItem{
		{
			OfComputerCall: &responses.ResponseComputerToolCallParam{
				ID:   "comp1",
				Type: responses.ResponseComputerToolCallTypeComputerCall,
				Action: responses.ResponseComputerToolCallActionUnionParam{
					OfScreenshot: &responses.ResponseComputerToolCallActionScreenshotParam{
						Type: constant.ValueOf[constant.Screenshot](),
					},
				},
				CallID:              "comp1",
				PendingSafetyChecks: nil,
				Status:              "completed",
			},
		},
	}, inputItems)
}

func TestToInputItemsForReasoning(t *testing.T) {
	// A reasoning output should produce the same dict as a reasoning input item.

	rc := responses.ResponseReasoningItemSummary{
		Text: "why",
		Type: constant.ValueOf[constant.SummaryText](),
	}
	reasoning := responses.ResponseOutputItemUnion{ // responses.ResponseReasoningItem
		ID:      "rid1",
		Summary: []responses.ResponseReasoningItemSummary{rc},
		Type:    "reasoning",
	}
	resp := agents.ModelResponse{
		Output:     []agents.TResponseOutputItem{reasoning},
		Usage:      usage.NewUsage(),
		ResponseID: "",
	}

	inputItems := resp.ToInputItems()

	// The value should contain exactly the primitive values of the message
	assert.Equal(t, []agents.TResponseInputItem{
		{
			OfReasoning: &responses.ResponseReasoningItemParam{
				ID: "rid1",
				Summary: []responses.ResponseReasoningItemSummaryParam{
					{
						Text: "why",
						Type: constant.ValueOf[constant.SummaryText](),
					},
				},
				Status: "",
				Type:   constant.ValueOf[constant.Reasoning](),
			},
		},
	}, inputItems)
}
