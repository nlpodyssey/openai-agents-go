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

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

/*
This example demonstrates a deterministic flow, where each step is performed by an agent.
1. The first agent generates a story outline
2. We feed the outline into the second agent
3. The second agent checks if the outline is good quality and if it is a scifi story
4. If the outline is not good quality or not a scifi story, we stop here
5. If the outline is good quality and a scifi story, we feed the outline into the third agent
6. The third agent writes the story
*/

const Model = "gpt-4o-mini"

var StoryOutlineAgent = agents.New("story_outline_agent").
	WithInstructions("Generate a very short story outline based on the user's input.").
	WithModel(Model)

type OutlineCheckerOutput struct {
	GoodQuality bool `json:"good_quality"`
	IsSciFi     bool `json:"is_scifi"`
}

type OutlineCheckerOutputSchema struct{}

func (s OutlineCheckerOutputSchema) Name() string             { return "OutlineCheckerOutput" }
func (s OutlineCheckerOutputSchema) IsPlainText() bool        { return false }
func (s OutlineCheckerOutputSchema) IsStrictJSONSchema() bool { return true }
func (s OutlineCheckerOutputSchema) JSONSchema() map[string]any {
	return map[string]any{
		"title":                "OutlineCheckerOutput",
		"type":                 "object",
		"required":             []string{"good_quality", "is_scifi"},
		"additionalProperties": false,
		"properties": map[string]any{
			"good_quality": map[string]any{
				"title": "Good Quality",
				"type":  "boolean",
			},
			"is_scifi": map[string]any{
				"title": "Is Scifi",
				"type":  "boolean",
			},
		},
	}
}
func (s OutlineCheckerOutputSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var v OutlineCheckerOutput
	err := dec.Decode(&v)
	return v, err
}

var OutlineCheckerAgent = agents.New("outline_checker_agent").
	WithInstructions("Read the given story outline, and judge the quality. Also, determine if it is a scifi story.").
	WithOutputSchema(OutlineCheckerOutputSchema{}).
	WithModel(Model)

var StoryAgent = agents.New("story_agent").
	WithInstructions("Write a short story based on the given outline.").
	WithModel(Model)

func main() {
	fmt.Print("What kind of story do you want? ")
	_ = os.Stdout.Sync()

	line, _, err := bufio.NewReader(os.Stdin).ReadLine()
	if err != nil {
		panic(err)
	}
	inputPrompt := string(line)

	// 1. Generate an outline
	outlineResult, err := agents.Run(context.Background(), StoryOutlineAgent, inputPrompt)
	if err != nil {
		panic(err)
	}

	fmt.Println("Outline generated")

	// 2. Check the outline
	outlineCheckerRunResult, err := agents.Run(context.Background(), OutlineCheckerAgent, outlineResult.FinalOutput.(string))
	if err != nil {
		panic(err)
	}

	// 3. Add a gate to stop if the outline is not good quality or not a scifi story
	outlineCheckerResult := outlineCheckerRunResult.FinalOutput.(OutlineCheckerOutput)
	if !outlineCheckerResult.GoodQuality {
		fmt.Println("Outline is not good quality, so we stop here.")
		return
	}
	if !outlineCheckerResult.IsSciFi {
		fmt.Println("Outline is not a scifi story, so we stop here.")
		return
	}
	fmt.Println("Outline is good quality and a scifi story, so we continue to write the story.")

	// 4. Write the story
	storyResult, err := agents.Run(context.Background(), StoryAgent, outlineResult.FinalOutput.(string))
	if err != nil {
		panic(err)
	}

	fmt.Printf("Story: %s\n", storyResult.FinalOutput)
}
