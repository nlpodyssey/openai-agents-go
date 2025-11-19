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
	"fmt"
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

/*
This example shows the LLM as a judge pattern. The first agent generates an outline for a story.
The second agent judges the outline and provides feedback. We loop until the judge is satisfied
with the outline.
*/

var StoryOutlineGenerator = agents.New("story_outline_generator").
	WithInstructions("You generate a very short story outline based on the user's input. If there is any feedback provided, use it to improve the outline.").
	WithModel("gpt-4.1-nano")

type EvaluationFeedback struct {
	Feedback string        `json:"feedback"`
	Score    FeedbackScore `json:"score" jsonschema:"enum=pass,enum=needs_improvement,enum=fail"`
}

type FeedbackScore string

const (
	FeedbackScorePass             FeedbackScore = "pass"
	FeedbackScoreNeedsImprovement FeedbackScore = "needs_improvement"
	FeedbackScoreFail             FeedbackScore = "fail"
)

var Evaluator = agents.New("evaluator").
	WithInstructions(
		"You evaluate a story outline and decide if it's good enough. " +
			"If it's not good enough, you provide feedback on what needs to be improved. " +
			"Never give it a pass on the first try. " +
			"After 5 attempts, you can give it a pass if the story outline is good enough - do not go for perfection.",
	).
	WithOutputType(agents.OutputType[EvaluationFeedback]()).
	WithModel("gpt-4.1-nano")

func main() {
	fmt.Print("What kind of story would you like to hear? ")
	_ = os.Stdout.Sync()
	line, _, err := bufio.NewReader(os.Stdin).ReadLine()
	if err != nil {
		panic(err)
	}
	msg := string(line)

	inputItems := []agents.TResponseInputItem{
		{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt(msg),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		},
	}
	var latestOutline *string

	// We'll run the entire workflow in a single trace
	err = tracing.RunTrace(
		context.Background(),
		tracing.TraceParams{WorkflowName: "LLM as a judge"},
		func(ctx context.Context, _ tracing.Trace) error {
			for {
				storyOutlineResult, err := agents.RunInputs(ctx, StoryOutlineGenerator, inputItems)
				if err != nil {
					return err
				}

				inputItems = storyOutlineResult.ToInputList()
				textMessageOutputs := agents.ItemHelpers().TextMessageOutputs(storyOutlineResult.NewItems)
				latestOutline = &textMessageOutputs

				fmt.Println("Story outline generated")

				evaluatorResult, err := agents.RunInputs(ctx, Evaluator, inputItems)
				if err != nil {
					return err
				}

				result := evaluatorResult.FinalOutput.(EvaluationFeedback)

				fmt.Printf("Evaluator score: %s\n", result.Score)

				if result.Score == FeedbackScorePass {
					fmt.Println("Story outline is good enough, exiting.")
					break
				}

				fmt.Println("Re-running with feedback")

				inputItems = append(inputItems, agents.TResponseInputItem{
					OfMessage: &responses.EasyInputMessageParam{
						Content: responses.EasyInputMessageContentUnionParam{
							OfString: param.NewOpt(fmt.Sprintf("Feedback: %s", result.Feedback)),
						},
						Role: responses.EasyInputMessageRoleUser,
						Type: responses.EasyInputMessageTypeMessage,
					},
				})
			}
			return nil
		},
	)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Final story outline: %s\n", *latestOutline)
}
