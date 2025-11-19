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

package agentstesting

import (
	"context"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
)

func GetTextInputItem(content string) agents.TResponseInputItem {
	return agents.TResponseInputItem{
		OfMessage: &responses.EasyInputMessageParam{
			Content: responses.EasyInputMessageContentUnionParam{
				OfString: param.NewOpt(content),
			},
			Role: responses.EasyInputMessageRoleUser,
			Type: responses.EasyInputMessageTypeMessage,
		},
	}
}

func GetTextMessage(content string) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{ // responses.ResponseOutputMessage
		ID:   "1",
		Type: "message",
		Role: constant.ValueOf[constant.Assistant](),
		Content: []responses.ResponseOutputMessageContentUnion{{ // responses.ResponseOutputText
			Text:        content,
			Type:        "output_text",
			Annotations: nil,
		}},
		Status: string(responses.ResponseOutputMessageStatusCompleted),
	}
}

func GetFunctionTool(name string, returnValue string) agents.FunctionTool {
	return agents.FunctionTool{
		Name: name,
		ParamsJSONSchema: map[string]any{
			"title":                name + "_args",
			"type":                 "object",
			"required":             []string{},
			"additionalProperties": false,
			"properties":           map[string]any{},
		},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return returnValue, nil
		},
	}
}

func GetFunctionToolErr(name string, returnErr error) agents.FunctionTool {
	return agents.FunctionTool{
		Name: name,
		ParamsJSONSchema: map[string]any{
			"title":                name + "_args",
			"type":                 "object",
			"required":             []string{},
			"additionalProperties": false,
			"properties":           map[string]any{},
		},
		OnInvokeTool: func(context.Context, string) (any, error) {
			return nil, returnErr
		},
	}
}

func GetFunctionToolCall(name string, arguments string) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{ // responses.ResponseFunctionToolCall
		ID:        "1",
		CallID:    "2",
		Type:      "function_call",
		Name:      name,
		Arguments: arguments,
	}
}

func GetHandoffToolCall(
	toAgent *agents.Agent,
	overrideName string,
	args string,
) responses.ResponseOutputItemUnion {
	name := overrideName
	if name == "" {
		name = agents.DefaultHandoffToolName(toAgent)
	}
	return GetFunctionToolCall(name, args)
}

func GetFinalOutputMessage(args string) responses.ResponseOutputItemUnion {
	return responses.ResponseOutputItemUnion{ // responses.ResponseOutputMessage
		ID:   "1",
		Type: "message",
		Role: constant.ValueOf[constant.Assistant](),
		Content: []responses.ResponseOutputMessageContentUnion{{ // responses.ResponseOutputText
			Text:        args,
			Type:        "output_text",
			Annotations: nil,
		}},
		Status: string(responses.ResponseOutputMessageStatusCompleted),
	}
}
