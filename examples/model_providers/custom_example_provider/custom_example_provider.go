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
	"fmt"
	"os"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/nlpodyssey/openai-agents-go/types/optional"
	"github.com/openai/openai-go/option"
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
This example uses a custom provider for some calls to Runner().Run(), and direct calls to OpenAI for
others. Steps:
1. Create a custom OpenAI client.
2. Create a ModelProvider that uses the custom client.
3. Use the ModelProvider in calls to Runner().Run(), only when we want to use the custom LLM provider.
*/

var Client = agents.NewOpenaiClient(
	optional.Value(BaseURL),
	option.WithAPIKey(APIKey),
)

type CustomModelProviderType struct{}

func (CustomModelProviderType) GetModel(modelName string) (agents.Model, error) {
	if modelName == "" {
		modelName = ModelName
	}
	return agents.NewOpenAIChatCompletionsModel(modelName, Client), nil
}

var CustomModelProvider = CustomModelProviderType{}

type GetWeatherArgs struct {
	City string `json:"city"`
}

func GetWeather(args GetWeatherArgs) string {
	fmt.Printf("[debug] getting weather for %s\n", args.City)
	return fmt.Sprintf("The weather in %s is sunny.", args.City)
}

var GetWeatherTool = agents.FunctionTool{
	Name:        "get_weather",
	Description: "",
	ParamsJSONSchema: map[string]any{
		"title":                "get_weather_args",
		"type":                 "object",
		"required":             []string{"city"},
		"additionalProperties": false,
		"properties": map[string]any{
			"city": map[string]any{
				"title": "City",
				"type":  "string",
			},
		},
	},
	OnInvokeTool: func(_ context.Context, _ *runcontext.RunContextWrapper, arguments string) (any, error) {
		var args GetWeatherArgs
		err := json.Unmarshal([]byte(arguments), &args)
		if err != nil {
			return nil, err
		}
		return GetWeather(args), nil
	},
}

func main() {

	agent := &agents.Agent{
		Name:         "Assistant",
		Instructions: agents.StringInstructions("You only respond in haikus."),
		Tools:        []agents.Tool{GetWeatherTool},
	}

	// This will use the custom model provider
	result, err := agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("What's the weather in Tokyo?"),
		RunConfig: optional.Value(agents.RunConfig{
			ModelProvider: CustomModelProvider,
		}),
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
}
