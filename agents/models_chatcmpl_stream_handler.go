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
	"reflect"

	"github.com/nlpodyssey/openai-agents-go/openaitypes"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

type StreamingState struct {
	Started                      bool
	TextContentIndexAndOutput    *textContentIndexAndOutput
	RefusalContentIndexAndOutput *refusalContentIndexAndOutput
	FunctionCalls                map[int64]*responses.ResponseOutputItemUnion // responses.ResponseFunctionToolCall
}

func NewStreamingState() StreamingState {
	return StreamingState{
		Started:                      false,
		TextContentIndexAndOutput:    nil,
		RefusalContentIndexAndOutput: nil,
		FunctionCalls:                make(map[int64]*responses.ResponseOutputItemUnion), // responses.ResponseFunctionToolCall
	}
}

type textContentIndexAndOutput struct {
	Index  int64
	Output responses.ResponseStreamEventUnionPart // responses.ResponseOutputText
}

type refusalContentIndexAndOutput struct {
	Index  int64
	Output responses.ResponseStreamEventUnionPart // responses.ResponseOutputRefusal
}

type SequenceNumber struct {
	n int64
}

func (sn *SequenceNumber) GetAndIncrement() int64 {
	n := sn.n
	sn.n += 1
	return n
}

type chatCmplStreamHandler struct{}

func ChatCmplStreamHandler() chatCmplStreamHandler { return chatCmplStreamHandler{} }

func (chatCmplStreamHandler) HandleStream(
	response responses.Response,
	stream *ssestream.Stream[openai.ChatCompletionChunk],
	yield func(TResponseStreamEvent) error,
) (err error) {
	defer func() {
		if e := stream.Close(); e != nil {
			err = errors.Join(err, fmt.Errorf("error closing stream: %w", e))
		}
	}()

	var completionUsage *openai.CompletionUsage
	state := NewStreamingState()
	sequenceNumber := SequenceNumber{}

	for stream.Next() {
		chunk := stream.Current()

		if !state.Started {
			state.Started = true
			if err = yield(TResponseStreamEvent{
				Response:       response,
				Type:           "response.created",
				SequenceNumber: sequenceNumber.GetAndIncrement(),
			}); err != nil {
				return err
			}
		}

		// This is always set by the OpenAI API, but not by others
		if !reflect.ValueOf(chunk.Usage).IsZero() {
			completionUsage = &chunk.Usage
		}

		if len(chunk.Choices) == 0 || reflect.ValueOf(chunk.Choices[0].Delta).IsZero() {
			continue
		}

		delta := chunk.Choices[0].Delta

		// Handle text
		if delta.Content != "" {
			if state.TextContentIndexAndOutput == nil {
				// Initialize a content tracker for streaming text
				state.TextContentIndexAndOutput = &textContentIndexAndOutput{
					Index: 0,
					Output: responses.ResponseStreamEventUnionPart{ // responses.ResponseOutputText
						Text:        "",
						Type:        "output_text",
						Annotations: nil,
					},
				}
				if state.RefusalContentIndexAndOutput != nil {
					state.TextContentIndexAndOutput.Index = 1
				}
				// Start a new assistant message stream
				assistantItem := responses.ResponseOutputItemUnion{ // responses.ResponseOutputMessage
					ID:      FakeResponsesID,
					Content: nil,
					Role:    constant.ValueOf[constant.Assistant](),
					Status:  string(responses.ResponseOutputMessageStatusInProgress),
					Type:    "message",
				}
				// Notify consumers of the start of a new output message + first content part
				if err = yield(TResponseStreamEvent{ // responses.ResponseOutputItemAddedEvent
					Item:           assistantItem,
					OutputIndex:    0,
					Type:           "response.output_item.added",
					SequenceNumber: sequenceNumber.GetAndIncrement(),
				}); err != nil {
					return err
				}
				if err = yield(TResponseStreamEvent{ // responses.ResponseContentPartAddedEvent
					ContentIndex: state.TextContentIndexAndOutput.Index,
					ItemID:       FakeResponsesID,
					OutputIndex:  0,
					Part: responses.ResponseStreamEventUnionPart{ // responses.ResponseOutputText
						Text:        "",
						Type:        "output_text",
						Annotations: nil,
					},
					Type:           "response.content_part.added",
					SequenceNumber: sequenceNumber.GetAndIncrement(),
				}); err != nil {
					return err
				}
			}
			// Emit the delta for this segment of content
			if err = yield(TResponseStreamEvent{ // responses.ResponseTextDeltaEvent
				ContentIndex:   state.TextContentIndexAndOutput.Index,
				Delta:          delta.Content,
				ItemID:         FakeResponsesID,
				OutputIndex:    0,
				Type:           "response.output_text.delta",
				SequenceNumber: sequenceNumber.GetAndIncrement(),
			}); err != nil {
				return err
			}
			// Accumulate the text into the response part
			state.TextContentIndexAndOutput.Output.Text += delta.Content
		}

		// Handle refusals (model declines to answer)
		// This is always set by the OpenAI API, but not by others
		if delta.Refusal != "" {
			if state.RefusalContentIndexAndOutput == nil {
				// Initialize a content tracker for streaming refusal text
				state.RefusalContentIndexAndOutput = &refusalContentIndexAndOutput{
					Index: 0,
					Output: responses.ResponseStreamEventUnionPart{ // responses.ResponseOutputRefusal
						Refusal: "",
						Type:    "refusal",
					},
				}
				if state.TextContentIndexAndOutput != nil {
					state.RefusalContentIndexAndOutput.Index = 1
				}
				// Start a new assistant message if one doesn't exist yet (in-progress)
				assistantItem := responses.ResponseOutputItemUnion{ // responses.ResponseOutputMessage
					ID:      FakeResponsesID,
					Content: nil,
					Role:    constant.ValueOf[constant.Assistant](),
					Status:  string(responses.ResponseOutputMessageStatusInProgress),
					Type:    "message",
				}
				// Notify downstream that assistant message + first content part are starting
				if err = yield(TResponseStreamEvent{ // responses.ResponseOutputItemAddedEvent
					Item:           assistantItem,
					OutputIndex:    0,
					Type:           "response.output_item.added",
					SequenceNumber: sequenceNumber.GetAndIncrement(),
				}); err != nil {
					return err
				}
				if err = yield(TResponseStreamEvent{ // responses.ResponseContentPartAddedEvent
					ContentIndex: state.RefusalContentIndexAndOutput.Index,
					ItemID:       FakeResponsesID,
					OutputIndex:  0,
					Part: responses.ResponseStreamEventUnionPart{ // responses.ResponseOutputText
						Text:        "",
						Type:        "output_text",
						Annotations: nil,
					},
					Type:           "response.content_part.added",
					SequenceNumber: sequenceNumber.GetAndIncrement(),
				}); err != nil {
					return err
				}
			}
			// Emit the delta for this segment of refusal
			if err = yield(TResponseStreamEvent{ // responses.ResponseRefusalDeltaEvent
				ContentIndex:   state.RefusalContentIndexAndOutput.Index,
				Delta:          delta.Refusal,
				ItemID:         FakeResponsesID,
				OutputIndex:    0,
				Type:           "response.refusal.delta",
				SequenceNumber: sequenceNumber.GetAndIncrement(),
			}); err != nil {
				return err
			}
			// Accumulate the refusal string in the output part
			state.RefusalContentIndexAndOutput.Output.Refusal += delta.Refusal
		}

		// Handle tool calls
		// Because we don't know the name of the function until the end of the stream, we'll
		// save everything and yield events at the end
		for _, tcDelta := range delta.ToolCalls {
			tc, ok := state.FunctionCalls[tcDelta.Index]
			if !ok {
				tc = &responses.ResponseOutputItemUnion{ // responses.ResponseFunctionToolCall
					ID:        FakeResponsesID,
					Arguments: "",
					Name:      "",
					Type:      "function_call",
					CallID:    "",
				}
				state.FunctionCalls[tcDelta.Index] = tc
			}
			tcFunction := tcDelta.Function

			tc.Arguments += tcFunction.Arguments
			tc.Name += tcFunction.Name
			if len(tcDelta.ID) > 0 {
				tc.CallID = tcDelta.ID
			}
		}
	}

	if err = stream.Err(); err != nil {
		return fmt.Errorf("error streaming response: %w", err)
	}

	functionCallStartingIndex := int64(0)
	if state.TextContentIndexAndOutput != nil {
		functionCallStartingIndex += 1
		// Send end event for this content part
		if err = yield(TResponseStreamEvent{ // responses.ResponseContentPartDoneEvent
			ContentIndex:   state.TextContentIndexAndOutput.Index,
			ItemID:         FakeResponsesID,
			OutputIndex:    0,
			Part:           state.TextContentIndexAndOutput.Output,
			Type:           "response.content_part.done",
			SequenceNumber: sequenceNumber.GetAndIncrement(),
		}); err != nil {
			return err
		}
	}

	if state.RefusalContentIndexAndOutput != nil {
		functionCallStartingIndex += 1
		// Send end event for this content part
		if err = yield(TResponseStreamEvent{ // responses.ResponseContentPartDoneEvent
			ContentIndex:   state.RefusalContentIndexAndOutput.Index,
			ItemID:         FakeResponsesID,
			OutputIndex:    0,
			Part:           state.RefusalContentIndexAndOutput.Output,
			Type:           "response.content_part.done",
			SequenceNumber: sequenceNumber.GetAndIncrement(),
		}); err != nil {
			return err
		}
	}

	// Actually send events for the function calls
	for _, functionCall := range state.FunctionCalls {
		// First, a ResponseOutputItemAdded for the function call
		if err = yield(TResponseStreamEvent{ // responses.ResponseOutputItemAddedEvent
			Item: responses.ResponseOutputItemUnion{ // responses.ResponseFunctionToolCall
				ID:        FakeResponsesID,
				CallID:    functionCall.CallID,
				Arguments: functionCall.Arguments,
				Name:      functionCall.Name,
				Type:      "function_call",
			},
			OutputIndex:    functionCallStartingIndex,
			Type:           "response.output_item.added",
			SequenceNumber: sequenceNumber.GetAndIncrement(),
		}); err != nil {
			return err
		}
		// Then, yield the args
		if err = yield(TResponseStreamEvent{ // responses.ResponseFunctionCallArgumentsDeltaEvent
			Delta:          functionCall.Arguments,
			ItemID:         FakeResponsesID,
			OutputIndex:    functionCallStartingIndex,
			Type:           "response.function_call_arguments.delta",
			SequenceNumber: sequenceNumber.GetAndIncrement(),
		}); err != nil {
			return err
		}
		// Finally, the ResponseOutputItemDone
		if err = yield(TResponseStreamEvent{ // responses.ResponseOutputItemDoneEvent
			Item: responses.ResponseOutputItemUnion{ // responses.ResponseFunctionToolCall
				ID:        FakeResponsesID,
				CallID:    functionCall.CallID,
				Arguments: functionCall.Arguments,
				Name:      functionCall.Name,
				Type:      "function_call",
			},
			OutputIndex:    functionCallStartingIndex,
			Type:           "response.output_item.done",
			SequenceNumber: sequenceNumber.GetAndIncrement(),
		}); err != nil {
			return err
		}
	}

	// Finally, send the Response completed event\
	var outputs []responses.ResponseOutputItemUnion
	if state.TextContentIndexAndOutput != nil || state.RefusalContentIndexAndOutput != nil {
		assistantMsg := responses.ResponseOutputItemUnion{ // responses.ResponseOutputMessage
			ID:      FakeResponsesID,
			Content: nil,
			Role:    constant.ValueOf[constant.Assistant](),
			Type:    "message",
			Status:  "completed",
		}
		if state.TextContentIndexAndOutput != nil {
			assistantMsg.Content = append(
				assistantMsg.Content,
				openaitypes.ResponseOutputMessageContentUnionFromResponseStreamEventUnionPart(
					state.TextContentIndexAndOutput.Output,
				),
			)
		}
		if state.RefusalContentIndexAndOutput != nil {
			assistantMsg.Content = append(
				assistantMsg.Content,
				openaitypes.ResponseOutputMessageContentUnionFromResponseStreamEventUnionPart(
					state.RefusalContentIndexAndOutput.Output,
				),
			)
		}
		outputs = append(outputs, assistantMsg)

		// send a ResponseOutputItemDone for the assistant message
		if err = yield(TResponseStreamEvent{ // responses.ResponseOutputItemDoneEvent
			Item:           assistantMsg,
			OutputIndex:    0,
			Type:           "response.output_item.done",
			SequenceNumber: sequenceNumber.GetAndIncrement(),
		}); err != nil {
			return err
		}
	}

	for _, functionCall := range state.FunctionCalls {
		outputs = append(outputs, *functionCall)
	}

	finalResponse := response // copy
	finalResponse.Output = outputs

	finalResponse.Usage = responses.ResponseUsage{}
	if completionUsage != nil {
		finalResponse.Usage = responses.ResponseUsage{
			InputTokens:  completionUsage.PromptTokens,
			OutputTokens: completionUsage.CompletionTokens,
			TotalTokens:  completionUsage.TotalTokens,
			OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
				ReasoningTokens: completionUsage.CompletionTokensDetails.ReasoningTokens,
			},
			InputTokensDetails: responses.ResponseUsageInputTokensDetails{
				CachedTokens: completionUsage.PromptTokensDetails.CachedTokens,
			},
		}
	}

	return yield(TResponseStreamEvent{ // responses.ResponseCompletedEvent
		Response:       finalResponse,
		Type:           "response.completed",
		SequenceNumber: sequenceNumber.GetAndIncrement(),
	})
}
