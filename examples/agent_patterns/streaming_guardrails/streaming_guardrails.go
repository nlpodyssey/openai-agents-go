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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/asynctask"
	"github.com/openai/openai-go/packages/param"
)

/*
This example shows how to use guardrails as the model is streaming. Output guardrails run after the
final output has been generated; this example runs guardrails every N tokens, allowing for early
termination if bad output is detected.

The expected output is that you'll see a bunch of tokens stream in, then the guardrail will trigger
and stop the streaming.
*/

var Model = agents.NewAgentModelName("gpt-4.1-nano")

var Agent = agents.New("Assistant").
	WithInstructions("You are a helpful assistant. You ALWAYS write long responses, making sure to be verbose and detailed.").
	WithModelOpt(param.NewOpt(Model))

type GuardrailOutput struct {
	// Reasoning about whether the response could be understood by a ten year old.
	Reasoning string `json:"reasoning"`

	// Whether the response is understandable by a ten year old.
	IsReadableByTenYearOld bool `json:"is_readable_by_ten_year_old"`
}

type GuardrailOutputSchema struct{}

func (s GuardrailOutputSchema) Name() string             { return "GuardrailOutput" }
func (s GuardrailOutputSchema) IsPlainText() bool        { return false }
func (s GuardrailOutputSchema) IsStrictJSONSchema() bool { return true }
func (s GuardrailOutputSchema) JSONSchema() map[string]any {
	return map[string]any{
		"title":                "GuardrailOutput",
		"type":                 "object",
		"required":             []string{"reasoning", "is_readable_by_ten_year_old"},
		"additionalProperties": false,
		"properties": map[string]any{
			"reasoning": map[string]any{
				"title":       "Reasoning",
				"description": "Reasoning about whether the response could be understood by a ten year old.",
				"type":        "string",
			},
			"is_readable_by_ten_year_old": map[string]any{
				"title":       "Is readable by ten year old",
				"description": "Whether the response is understandable by a ten year old.",
				"type":        "boolean",
			},
		},
	}
}
func (s GuardrailOutputSchema) ValidateJSON(jsonStr string) (any, error) {
	r := strings.NewReader(jsonStr)
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()

	var v GuardrailOutput
	err := dec.Decode(&v)
	return v, err
}

var GuardrailAgent = agents.New("Checker").
	WithInstructions("You will be given a question and a response. Your goal is to judge whether the response is simple enough to be understood by a ten year old.").
	WithOutputSchema(GuardrailOutputSchema{}).
	WithModelOpt(param.NewOpt(Model))

type CheckGuardrailResult struct {
	Output GuardrailOutput
	Error  error
}

func CheckGuardrail(text string) CheckGuardrailResult {
	result, err := agents.Runner{}.Run(context.Background(), GuardrailAgent, agents.InputString(text))
	if err != nil {
		return CheckGuardrailResult{Error: err}
	}
	return CheckGuardrailResult{Output: result.FinalOutput.(GuardrailOutput)}
}

func main() {
	question := "What is a black hole, and how does it behave?"
	result, err := agents.Runner{}.RunStreamed(
		context.Background(), Agent, agents.InputString(question),
	)
	if err != nil {
		panic(err)
	}

	currentText := ""

	// We will check the guardrail every N characters
	nextGuardrailCheckLen := 300
	var guardrailTask *asynctask.Task[CheckGuardrailResult]

	streamBreakErr := errors.New("stream break")

	err = result.StreamEvents(func(event agents.StreamEvent) error {
		if e, ok := event.(agents.RawResponsesStreamEvent); ok && e.Data.Type == "response.output_text.delta" {
			fmt.Print(e.Data.Delta.OfString)
			_ = os.Stdout.Sync()
			currentText += e.Data.Delta.OfString

			// Check if it's time to run the guardrail check
			// Note that we don't run the guardrail check if there's already a task running. An
			// alternate implementation is to have N guardrails running, or cancel the previous
			// one.
			if len(currentText) >= nextGuardrailCheckLen && guardrailTask == nil {
				fmt.Println("Running guardrail check")
				currentText := currentText
				guardrailTask = asynctask.CreateTask(context.Background(), func(context.Context) CheckGuardrailResult {
					return CheckGuardrail(currentText)
				})
				nextGuardrailCheckLen += 300
			}
		}

		// Every iteration of the loop, check if the guardrail has been triggered
		if guardrailTask != nil && guardrailTask.IsDone() {
			taskResult := guardrailTask.Await()
			if taskResult.Canceled {
				return errors.New("guardrail result canceled")
			}
			if err := taskResult.Result.Error; err != nil {
				return err
			}
			guardrailResult := taskResult.Result.Output
			if !guardrailResult.IsReadableByTenYearOld {
				fmt.Print("\n\n================\n\n\n")
				fmt.Printf("Guardrail triggered. Reasoning:\n%s\n", guardrailResult.Reasoning)
				return streamBreakErr
			}
		}

		return nil
	})
	if err != nil && !errors.Is(err, streamBreakErr) {
		panic(err)
	}

	// Do one final check on the final output
	checkResult := CheckGuardrail(currentText)
	if err = checkResult.Error; err != nil {
		panic(err)
	}
	guardrailResult := checkResult.Output
	if !guardrailResult.IsReadableByTenYearOld {
		fmt.Print("\n\n================\n\n\n")
		fmt.Printf("Guardrail triggered. Reasoning:\n%s\n", guardrailResult.Reasoning)
	}
}
