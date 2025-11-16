// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agents

import (
	"context"

	"github.com/openai/openai-go/v3/responses"
)

// ToolContextData provides context data of a tool call.
type ToolContextData struct {
	// The name of the tool being invoked.
	ToolName string

	// The ID of the tool call.
	ToolCallID string
}

type toolContextDataKey struct{}

func ContextWithToolData(
	ctx context.Context,
	toolCallID string,
	toolCall responses.ResponseFunctionToolCall,
) context.Context {
	return context.WithValue(ctx, toolContextDataKey{}, &ToolContextData{
		ToolName:   toolCall.Name,
		ToolCallID: toolCallID,
	})
}

func ToolDataFromContext(ctx context.Context) *ToolContextData {
	v, _ := ctx.Value(toolContextDataKey{}).(*ToolContextData)
	return v
}
