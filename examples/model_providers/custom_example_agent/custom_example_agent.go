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
	"fmt"
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
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
This example uses a custom provider for a specific agent. Steps:
1. Create a custom OpenAI client.
2. Create a `Model` that uses the custom client.
3. Set the `model` on the Agent.
*/

var Client = agents.NewOpenaiClient(
	param.NewOpt(BaseURL),
	option.WithAPIKey(APIKey),
)

// An alternate approach that would also work:
//
// provider := agents.NewOpenAIProvider(agents.OpenAIProviderParams{
//     OpenaiClient: &Client,
// })
// agent := agents.New("Assistant").
//         // ...
//         WithModel("some-custom-model")
// result, err := (agents.Runner{Config: agents.RunConfig{ModelProvider: provider}}).
//         Run(context.Background(), agent /* ... */)
//

type GetWeatherArgs struct {
	City string `json:"city"`
}

func GetWeather(_ context.Context, args GetWeatherArgs) (string, error) {
	fmt.Printf("[debug] getting weather for %s\n", args.City)
	return fmt.Sprintf("The weather in %s is sunny.", args.City), nil
}

var GetWeatherTool = agents.NewFunctionTool("get_weather", "", GetWeather)

func main() {
	// This agent will use the custom LLM provider
	agent := agents.New("Assistant").
		WithInstructions("You only respond in haikus.").
		WithModelInstance(agents.NewOpenAIChatCompletionsModel(ModelName, Client)).
		WithTools(GetWeatherTool)

	result, err := agents.Run(context.Background(), agent, "What's the weather in Tokyo?")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
}
