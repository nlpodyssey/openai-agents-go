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

package handoff_filters_test

import (
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agents/extensions/handoff_filters"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
)

func newFakeAgent() *agents.Agent {
	return &agents.Agent{
		Name: "fake_agent",
	}
}

func getMessageInputItem(content string) agents.TResponseInputItem {
	return agents.TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(content),
			},
			Role: responses.EasyInputMessageRoleAssistant,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
}

func getFunctionResultInputItem(content string) agents.TResponseInputItem {
	return agents.TResponseInputItem{
		OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
			CallID: "1",
			Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
				OfString: param.NewOpt(content),
			},
			Type: constant.ValueOf[constant.FunctionCallOutput](),
		},
	}
}

func getMessageOutputRunItem(content string) agents.MessageOutputItem {
	return agents.MessageOutputItem{
		Agent: newFakeAgent(),
		Type:  "message_output_item",
		RawItem: responses.ResponseOutputMessage{
			ID: "1",
			Content: []responses.ResponseOutputMessageContentUnion{{
				Text:        content,
				Annotations: nil,
				Type:        "output_text",
			}},
			Role:   constant.ValueOf[constant.Assistant](),
			Status: responses.ResponseOutputMessageStatusCompleted,
			Type:   constant.ValueOf[constant.Message](),
		},
	}
}

func getToolOutputRunItem(content string) agents.ToolCallOutputItem {
	return agents.ToolCallOutputItem{
		Agent: newFakeAgent(),
		RawItem: agents.ResponseInputItemFunctionCallOutputParam{
			CallID: "1",
			Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
				OfString: param.NewOpt(content),
			},
			Type: constant.ValueOf[constant.FunctionCallOutput](),
		},
		Output: content,
		Type:   "tool_call_output_item",
	}
}

func getHandoffInputItem(content string) agents.TResponseInputItem {
	return agents.TResponseInputItem{
		OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
			CallID: "1",
			Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
				OfString: param.NewOpt(content),
			},
			Type: constant.ValueOf[constant.FunctionCallOutput](),
		},
	}
}

func getHandoffOutputRunItem(content string) agents.HandoffOutputItem {
	return agents.HandoffOutputItem{
		Agent: newFakeAgent(),
		RawItem: agents.TResponseInputItem{
			OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
				CallID: "1",
				Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
					OfString: param.NewOpt(content),
				},
				Type: constant.ValueOf[constant.FunctionCallOutput](),
			},
		},
		SourceAgent: newFakeAgent(),
		TargetAgent: newFakeAgent(),
		Type:        "handoff_output_item",
	}
}

func TestEmptyData(t *testing.T) {
	handoffInputData := agents.HandoffInputData{
		InputHistory:    nil,
		PreHandoffItems: nil,
		NewItems:        nil,
	}
	filteredData := handoff_filters.RemoveAllTools(handoffInputData)
	assert.Equal(t, handoffInputData, filteredData)
}

func TestStringHistoryOnly(t *testing.T) {
	handoffInputData := agents.HandoffInputData{
		InputHistory:    agents.InputString("Hello"),
		PreHandoffItems: nil,
		NewItems:        nil,
	}
	filteredData := handoff_filters.RemoveAllTools(handoffInputData)
	assert.Equal(t, handoffInputData, filteredData)
}

func TestStringHistoryAndList(t *testing.T) {
	handoffInputData := agents.HandoffInputData{
		InputHistory:    agents.InputString("Hello"),
		PreHandoffItems: nil,
		NewItems: []agents.RunItem{
			getMessageOutputRunItem("Hello"),
		},
	}
	filteredData := handoff_filters.RemoveAllTools(handoffInputData)
	assert.Equal(t, handoffInputData, filteredData)
}

func TestListHistoryAndList(t *testing.T) {
	handoffInputData := agents.HandoffInputData{
		InputHistory: agents.InputItems{
			getMessageInputItem("Hello"),
		},
		PreHandoffItems: []agents.RunItem{
			getMessageOutputRunItem("123"),
		},
		NewItems: []agents.RunItem{
			getMessageOutputRunItem("World"),
		},
	}
	filteredData := handoff_filters.RemoveAllTools(handoffInputData)
	assert.Equal(t, handoffInputData, filteredData)
}

func TestRemovesToolsFromHistory(t *testing.T) {
	handoffInputData := agents.HandoffInputData{
		InputHistory: agents.InputItems{
			getMessageInputItem("Hello1"),
			getFunctionResultInputItem("World"),
			getMessageInputItem("Hello2"),
		},
		PreHandoffItems: []agents.RunItem{
			getToolOutputRunItem("abc"),
			getMessageOutputRunItem("123"),
		},
		NewItems: []agents.RunItem{
			getMessageOutputRunItem("World"),
		},
	}
	filteredData := handoff_filters.RemoveAllTools(handoffInputData)

	assert.Len(t, filteredData.InputHistory, 2)
	assert.Len(t, filteredData.PreHandoffItems, 1)
	assert.Len(t, filteredData.NewItems, 1)
}

func TestRemovesToolsFromNewItems(t *testing.T) {
	handoffInputData := agents.HandoffInputData{
		InputHistory:    nil,
		PreHandoffItems: nil,
		NewItems: []agents.RunItem{
			getMessageOutputRunItem("Hello"),
			getToolOutputRunItem("World"),
		},
	}
	filteredData := handoff_filters.RemoveAllTools(handoffInputData)

	assert.Nil(t, filteredData.InputHistory)
	assert.Nil(t, filteredData.PreHandoffItems)
	assert.Len(t, filteredData.NewItems, 1)
}

func TestRemovesToolsFromNewItemsAndHistory(t *testing.T) {
	handoffInputData := agents.HandoffInputData{
		InputHistory: agents.InputItems{
			getMessageInputItem("Hello1"),
			getFunctionResultInputItem("World"),
			getMessageInputItem("Hello2"),
		},
		PreHandoffItems: []agents.RunItem{
			getMessageOutputRunItem("123"),
			getToolOutputRunItem("456"),
		},
		NewItems: []agents.RunItem{
			getMessageOutputRunItem("Hello"),
			getToolOutputRunItem("World"),
		},
	}
	filteredData := handoff_filters.RemoveAllTools(handoffInputData)

	assert.Len(t, filteredData.InputHistory, 2)
	assert.Len(t, filteredData.PreHandoffItems, 1)
	assert.Len(t, filteredData.NewItems, 1)
}

func TestRemovesHandoffsFromHistory(t *testing.T) {
	handoffInputData := agents.HandoffInputData{
		InputHistory: agents.InputItems{
			getMessageInputItem("Hello1"),
			getHandoffInputItem("World"),
		},
		PreHandoffItems: []agents.RunItem{
			getMessageOutputRunItem("Hello"),
			getToolOutputRunItem("World"),
			getHandoffOutputRunItem("World"),
		},
		NewItems: []agents.RunItem{
			getMessageOutputRunItem("Hello"),
			getToolOutputRunItem("World"),
			getHandoffOutputRunItem("World"),
		},
	}
	filteredData := handoff_filters.RemoveAllTools(handoffInputData)

	assert.Len(t, filteredData.InputHistory, 1)
	assert.Len(t, filteredData.PreHandoffItems, 1)
	assert.Len(t, filteredData.NewItems, 1)
}
