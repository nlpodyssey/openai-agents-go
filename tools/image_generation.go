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

package tools

import (
	"context"
	"errors"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
)

// ImageGeneration is a tool that allows the LLM to generate images.
type ImageGeneration struct {
	// The tool config, which image generation settings.
	ToolConfig responses.ToolImageGenerationParam
}

func (fs ImageGeneration) ToolName() string {
	return "image_generation"
}

func (fs ImageGeneration) ConvertToResponses(context.Context) (*responses.ToolUnionParam, *responses.ResponseIncludable, error) {
	return &responses.ToolUnionParam{
		OfImageGeneration: &fs.ToolConfig,
	}, nil, nil
}

func (fs ImageGeneration) ConvertToChatCompletions(context.Context) (*openai.ChatCompletionToolParam, error) {
	return nil, errors.New("ImageGeneration.ConvertToChatCompletions not implemented")
}
