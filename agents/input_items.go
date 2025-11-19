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
	"fmt"

	"github.com/openai/openai-go/v3/responses"
)

// MessageItem returns a message input item with the given role and text.
func MessageItem(role responses.EasyInputMessageRole, text string) TResponseInputItem {
	return responses.ResponseInputItemParamOfMessage(text, role)
}

// UserMessage is a shorthand for a user role message.
func UserMessage(text string) TResponseInputItem {
	return MessageItem(responses.EasyInputMessageRoleUser, text)
}

// AssistantMessage is a shorthand for an assistant role message.
func AssistantMessage(text string) TResponseInputItem {
	return MessageItem(responses.EasyInputMessageRoleAssistant, text)
}

// SystemMessage is a shorthand for a system role message.
func SystemMessage(text string) TResponseInputItem {
	return MessageItem(responses.EasyInputMessageRoleSystem, text)
}

// DeveloperMessage is a shorthand for a developer role message.
func DeveloperMessage(text string) TResponseInputItem {
	return MessageItem(responses.EasyInputMessageRoleDeveloper, text)
}

// InputList builds a slice of input items from the provided values. Supported
// values are:
//   - string: converted to a user message
//   - TResponseInputItem: used as-is
//   - RunItem: converted via ToInputItem
//   - []TResponseInputItem: appended as-is
//   - []RunItem: each converted via ToInputItem
//   - ModelResponse or []ModelResponse: converted to their input items
//
// This allows passing already-constructed lists of
// [responses.ResponseInputItemUnionParam] (or the alias `TResponseInputItem`)
// directly when you have them available.
func InputList(values ...any) []TResponseInputItem {
	var out []TResponseInputItem
	for _, val := range values {
		switch v := val.(type) {
		case string:
			out = append(out, UserMessage(v))
		case TResponseInputItem:
			out = append(out, v)
		case RunItem:
			out = append(out, v.ToInputItem())
		case []TResponseInputItem:
			out = append(out, v...)
		case []RunItem:
			for _, item := range v {
				out = append(out, item.ToInputItem())
			}
		case ModelResponse:
			out = append(out, v.ToInputItems()...)
		case []ModelResponse:
			for _, mr := range v {
				out = append(out, mr.ToInputItems()...)
			}
		default:
			panic(fmt.Errorf("unsupported input value type %T", val))
		}
	}
	return out
}
