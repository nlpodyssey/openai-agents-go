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

// Package handoff_filters contains common handoff input filters, for convenience.
package handoff_filters

import (
	"fmt"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/v3/packages/param"
)

// RemoveAllTools filters out all tool items: file search, web search and function calls+output.
func RemoveAllTools(handoffInputData agents.HandoffInputData) agents.HandoffInputData {
	history := handoffInputData.InputHistory
	newItems := handoffInputData.NewItems

	var filteredHistory agents.Input
	switch history := history.(type) {
	case agents.InputString:
		filteredHistory = history
	case agents.InputItems:
		filteredHistory = agents.InputItems(removeToolTypesFromInput(history))
	default:
		if history != nil {
			// This would be an unrecoverable implementation bug, so a panic is appropriate.
			panic(fmt.Errorf("unexpected Input type %T", history))
		}
	}

	filteredPreHandoffItems := removeToolsFromItems(handoffInputData.PreHandoffItems)
	filteredNewItems := removeToolsFromItems(newItems)

	return agents.HandoffInputData{
		InputHistory:    filteredHistory,
		PreHandoffItems: filteredPreHandoffItems,
		NewItems:        filteredNewItems,
	}
}

func removeToolsFromItems(items []agents.RunItem) []agents.RunItem {
	if items == nil {
		return nil
	}
	filteredItems := make([]agents.RunItem, 0)
	for _, item := range items {
		switch item.(type) {
		case agents.HandoffCallItem, agents.HandoffOutputItem, agents.ToolCallItem, agents.ToolCallOutputItem:
			continue
		default:
			filteredItems = append(filteredItems, item)
		}
	}
	return filteredItems
}

func removeToolTypesFromInput(items []agents.TResponseInputItem) []agents.TResponseInputItem {
	filteredItems := make([]agents.TResponseInputItem, 0)
	for _, item := range items {
		if !param.IsOmitted(item.OfFunctionCall) || !param.IsOmitted(item.OfFunctionCallOutput) ||
			!param.IsOmitted(item.OfComputerCall) || !param.IsOmitted(item.OfComputerCallOutput) ||
			!param.IsOmitted(item.OfFileSearchCall) || !param.IsOmitted(item.OfWebSearchCall) {
			continue
		}
		filteredItems = append(filteredItems, item)
	}
	return filteredItems
}
