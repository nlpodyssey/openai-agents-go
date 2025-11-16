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
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/openaitypes"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"github.com/stretchr/testify/assert"
)

func TestMessageHelpers(t *testing.T) {
	assert.Equal(t,
		responses.ResponseInputItemParamOfMessage("foo", responses.EasyInputMessageRoleUser),
		agents.UserMessage("foo"),
	)
	assert.Equal(t,
		responses.ResponseInputItemParamOfMessage("bar", responses.EasyInputMessageRoleAssistant),
		agents.AssistantMessage("bar"),
	)
	assert.Equal(t,
		responses.ResponseInputItemParamOfMessage("sys", responses.EasyInputMessageRoleSystem),
		agents.SystemMessage("sys"),
	)
	assert.Equal(t,
		responses.ResponseInputItemParamOfMessage("dev", responses.EasyInputMessageRoleDeveloper),
		agents.DeveloperMessage("dev"),
	)
}

func TestInputList(t *testing.T) {
	msg := agents.UserMessage("hi")

	outMsgUnion := agentstesting.GetTextMessage("reply")
	runItem := agents.MessageOutputItem{
		Agent:   &agents.Agent{Name: "a"},
		RawItem: openaitypes.ResponseOutputMessageFromResponseOutputItemUnion(outMsgUnion),
		Type:    "message_output_item",
	}

	mr := agents.ModelResponse{Output: []agents.TResponseOutputItem{outMsgUnion}}

	result := agents.InputList(
		"hello", msg, runItem,
		[]agents.TResponseInputItem{msg},
		[]agents.RunItem{runItem},
		mr, []agents.ModelResponse{mr},
	)

	expected := []agents.TResponseInputItem{
		agents.UserMessage("hello"),
		msg,
		runItem.ToInputItem(),
		msg,
		runItem.ToInputItem(),
	}
	expected = append(expected, mr.ToInputItems()...)
	expected = append(expected, mr.ToInputItems()...)

	assert.Equal(t, expected, result)
}

func TestInputList_WithUnionItem(t *testing.T) {
	reasoning := responses.ResponseReasoningItem{
		ID: "rid1",
		Summary: []responses.ResponseReasoningItemSummary{{
			Text: "why",
			Type: constant.ValueOf[constant.SummaryText](),
		}},
		Type: constant.ValueOf[constant.Reasoning](),
	}

	item := openaitypes.ResponseInputItemUnionParamFromResponseReasoningItem(reasoning)

	result := agents.InputList(item)

	assert.Equal(t, []agents.TResponseInputItem{item}, result)
}
