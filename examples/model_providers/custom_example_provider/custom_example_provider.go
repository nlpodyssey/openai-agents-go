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
	"cmp"
	"context"
	"fmt"
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3/packages/param"
)

var (
	BaseURL   = os.Getenv("EXAMPLE_BASE_URL")
	APIKey    = os.Getenv("EXAMPLE_API_KEY")
	ModelName = os.Getenv("EXAMPLE_MODEL_NAME")
)

func init() {
	if BaseURL == "" || APIKey == "" || ModelName == "" {
		fmt.Println("Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code.")
		os.Exit(1)
	}
}

/*
This example uses a custom provider for some calls to Runner.Run(), and direct calls to OpenAI for
others. Steps:
1. Create a custom OpenAI client.
2. Create a ModelProvider that uses the custom client.
3. Use the ModelProvider in calls to Runner.Run(), only when we want to use the custom LLM provider.

Note that in this example, we disable tracing under the assumption that you don't have an API key
from platform.openai.com. If you do have one, you can either set the `OPENAI_API_KEY` env var
or call tracing.SetTracingExportAPIKey to set a tracing specific key.
*/

var Client = agents.NewOpenaiClient(param.NewOpt(BaseURL), param.NewOpt(APIKey))

func init() {
	tracing.SetTracingDisabled(true)
}

type CustomModelProviderType struct{}

func (CustomModelProviderType) GetModel(modelName string) (agents.Model, error) {
	return agents.NewOpenAIChatCompletionsModel(cmp.Or(modelName, ModelName), Client), nil
}

var CustomModelProvider = CustomModelProviderType{}

type GetWeatherArgs struct {
	City string `json:"city"`
}

func GetWeather(_ context.Context, args GetWeatherArgs) (string, error) {
	fmt.Printf("[debug] getting weather for %s\n", args.City)
	return fmt.Sprintf("The weather in %s is sunny.", args.City), nil
}

var GetWeatherTool = agents.NewFunctionTool("get_weather", "", GetWeather)

func main() {

	agent := agents.New("Assistant").
		WithInstructions("You only respond in haikus.").
		WithTools(GetWeatherTool)

	// This will use the custom model provider
	result, err := (agents.Runner{Config: agents.RunConfig{ModelProvider: CustomModelProvider}}).
		Run(context.Background(), agent, "What's the weather in Tokyo?")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
}
