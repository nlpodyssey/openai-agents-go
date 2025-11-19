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
	"log"
	"os"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared/constant"
)

/*
This example demonstrates how to use LiteLLM running locally in Docker as a model provider.

Prerequisites:
1. Docker with LiteLLM running on port 4000 (see README.md for setup)
2. OpenAI API key set in environment or docker container

The example shows three different approaches:
1. Direct LiteLLM provider with local Docker endpoint
2. Using MultiProvider with prefix routing (litellm/model-name)
3. Setting LiteLLM as the global default provider

Run this after starting your Docker container:
docker run -v $(pwd)/litellm_config.yaml:/app/config.yaml -e OPENAI_API_KEY="sk-..." -p 4000:4000 ghcr.io/berriai/litellm:main-stable --config /app/config.yaml --detailed_debug
*/

// LiteLLMProvider implements ModelProvider for local LiteLLM Docker instance
type LiteLLMProvider struct {
	baseURL string
	apiKey  string
	client  *agents.OpenaiClient
}

// NewLiteLLMProvider creates a provider that connects to local LiteLLM Docker instance
func NewLiteLLMProvider(baseURL, apiKey string) *LiteLLMProvider {
	if baseURL == "" {
		baseURL = "http://localhost:4000" // Default Docker port
	}
	if apiKey == "" {
		apiKey = "dummy-key" // LiteLLM proxy doesn't always require the actual key
	}

	client := agents.NewOpenaiClient(
		param.NewOpt(baseURL),
		param.NewOpt(apiKey),
	)

	return &LiteLLMProvider{
		baseURL: baseURL,
		apiKey:  apiKey,
		client:  &client,
	}
}

// GetModel returns a model that connects to LiteLLM
func (p *LiteLLMProvider) GetModel(modelName string) (agents.Model, error) {
	if modelName == "" {
		modelName = "openai-gpt-4o" // Default model from our config
	}

	log.Printf("Creating LiteLLM model: %s via %s", modelName, p.baseURL)
	return agents.NewOpenAIChatCompletionsModel(modelName, *p.client), nil
}

// Weather tool for demonstration
type GetWeatherArgs struct {
	City string `json:"city" description:"The city to get weather for"`
}

func GetWeather(ctx context.Context, args GetWeatherArgs) (string, error) {
	fmt.Printf("🌤️  [debug] Getting weather for %s\n", args.City)
	return fmt.Sprintf("The weather in %s is sunny with a temperature of 22°C.", args.City), nil
}

var GetWeatherTool = agents.NewFunctionTool("get_weather", "Get current weather for a city", GetWeather)

func init() {
	// Disable tracing for this example
	tracing.SetTracingDisabled(true)
}

func main() {
	fmt.Println("🚀 Starting LiteLLM Docker Integration Example")
	fmt.Println(strings.Repeat("=", 50))

	// Verify LiteLLM is running
	if !checkLiteLLMConnection() {
		fmt.Println("❌ LiteLLM Docker container not accessible on port 4000")
		fmt.Println("Please start LiteLLM Docker first:")
		fmt.Println("docker run -v $(pwd)/litellm_config.yaml:/app/config.yaml -e OPENAI_API_KEY=\"your-key\" -p 4000:4000 ghcr.io/berriai/litellm:main-stable --config /app/config.yaml")
		os.Exit(1)
	}

	fmt.Println("✅ LiteLLM Docker container is running")

	// Example 1: Direct LiteLLM Provider
	fmt.Println("\n📝 Example 1: Direct LiteLLM Provider")
	fmt.Println(strings.Repeat("-", 40))
	runDirectLiteLLMExample()

	// Example 2: MultiProvider with Prefix Routing
	fmt.Println("\n📝 Example 2: MultiProvider with Prefix Routing")
	fmt.Println(strings.Repeat("-", 40))
	runMultiProviderExample()

	// Example 3: Global Default Provider
	fmt.Println("\n📝 Example 3: Global Default Provider")
	fmt.Println(strings.Repeat("-", 40))
	runGlobalDefaultExample()

	fmt.Println("\n🎉 All examples completed successfully!")
}

// Example 1: Using LiteLLM provider directly
func runDirectLiteLLMExample() {
	litellmProvider := NewLiteLLMProvider("", "")

	agent := agents.New("Weather Assistant").
		WithInstructions("You are a helpful weather assistant. Always use the get_weather tool to get current weather data.").
		WithTools(GetWeatherTool)

	result, err := (agents.Runner{
		Config: agents.RunConfig{
			ModelProvider: litellmProvider,
			Model:         param.NewOpt(agents.NewAgentModelName("openai-gpt-4o")),
		},
	}).Run(context.Background(), agent, "What's the weather like in Tokyo?")

	if err != nil {
		log.Printf("❌ Error in direct provider example: %v", err)
		return
	}

	fmt.Printf("🤖 Response: %s\n", result.FinalOutput)
}

// Example 2: Using MultiProvider with prefix routing
func runMultiProviderExample() {
	// Setup providers
	providerMap := agents.NewMultiProviderMap()
	providerMap.AddProvider("litellm", NewLiteLLMProvider("", ""))

	multiProvider := agents.NewMultiProvider(agents.NewMultiProviderParams{
		ProviderMap: providerMap,
	})

	agent := agents.New("Travel Assistant").
		WithInstructions("You are a travel assistant. You respond in a friendly, enthusiastic way about travel destinations.").
		WithTools(GetWeatherTool)

	// Note the "litellm/" prefix in the model name
	result, err := (agents.Runner{
		Config: agents.RunConfig{
			ModelProvider: multiProvider,
			Model:         param.NewOpt(agents.NewAgentModelName("litellm/openai-gpt-4o")),
		},
	}).Run(context.Background(), agent, "I'm planning a trip to Paris. What should I know about the weather?")

	if err != nil {
		log.Printf("❌ Error in multi-provider example: %v", err)
		return
	}

	fmt.Printf("🤖 Response: %s\n", result.FinalOutput)
}

// Example 3: Setting LiteLLM as global default
func runGlobalDefaultExample() {
	// Set up LiteLLM as the global default client
	litellmClient := agents.NewOpenaiClient(
		param.NewOpt("http://localhost:4000"),
		param.NewOpt("dummy-key"),
	)

	// Store current defaults to restore later
	originalClient := agents.GetDefaultOpenaiClient()
	originalUseResponses := agents.GetUseResponsesByDefault()

	// Set LiteLLM as default
	agents.SetDefaultOpenaiClient(litellmClient, false)
	agents.SetDefaultOpenaiAPI(agents.OpenaiAPITypeChatCompletions)

	defer func() {
		// Restore original settings
		if originalClient != nil {
			agents.SetDefaultOpenaiClient(*originalClient, false)
		} else {
			agents.ClearOpenaiSettings()
		}
		agents.SetUseResponsesByDefault(originalUseResponses)
	}()

	agent := agents.New("Poet Assistant").
		WithInstructions("You are a creative poet. Always end your responses with a short haiku about the topic.").
		WithModel("openai-gpt-4o"). // No prefix needed when using global default
		WithTools(GetWeatherTool)

	result, err := agents.Run(context.Background(), agent, "Tell me about the weather in San Francisco and write a poem about it")

	if err != nil {
		log.Printf("❌ Error in global default example: %v", err)
		return
	}

	fmt.Printf("🤖 Response: %s\n", result.FinalOutput)
}

// Helper function to check if LiteLLM is accessible
func checkLiteLLMConnection() bool {
	client := agents.NewOpenaiClient(
		param.NewOpt("http://localhost:4000"),
		param.NewOpt("dummy-key"),
	)

	// Simple test request
	_, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model: "openai-gpt-4o",
		Messages: []openai.ChatCompletionMessageParamUnion{{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: param.NewOpt("test"),
				},
				Role: constant.ValueOf[constant.User](),
			},
		}},
		MaxTokens: param.NewOpt(int64(1)),
	})

	return err == nil
}
