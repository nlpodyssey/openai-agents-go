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
	"flag"
	"fmt"
	"syscall"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/openai/openai-go/v3/packages/param"
)

/*
This example shows how to force the agent to use a tool.
It uses ModelSettings ToolChoice "required" to force the agent to use any tool.

You can run it with 3 options:
1. `default`: The default behavior, which is to send the tool output to the LLM. In this case,
   `ToolChoice` is not set, because otherwise it would result in an infinite loop - the LLM would
   call the tool, the tool would run and send the results to the LLM, and that would repeat
   (because the model is forced to use a tool every time.)
2. `first_tool_result`: The first tool result is used as the final output.
3. `custom`: A custom tool use behavior function is used. The custom function receives all the tool
   results, and chooses to use the first tool result to generate the final output.

Usage:
./forcing_tool_use -t default
./forcing_tool_use -t first_tool
./forcing_tool_use -t custom
*/

type Weather struct {
	City             string `json:"city"`
	TemperatureRange string `json:"temperature_range"`
	Conditions       string `json:"conditions"`
}

type GetWeatherArgs struct {
	City string `json:"city"`
}

func GetWeather(_ context.Context, args GetWeatherArgs) (Weather, error) {
	fmt.Println("[debug] GetWeather called")
	return Weather{
		City:             args.City,
		TemperatureRange: "14-20C",
		Conditions:       "Sunny with wind",
	}, nil
}

var GetWeatherTool = agents.NewFunctionTool("get_weather", "", GetWeather)

func CustomToolUseBehavior(_ context.Context, results []agents.FunctionToolResult) (agents.ToolsToFinalOutputResult, error) {
	weather := results[0].Output.(Weather)
	return agents.ToolsToFinalOutputResult{
		IsFinalOutput: true,
		FinalOutput: param.NewOpt[any](
			fmt.Sprintf("%s is %s", weather.City, weather.Conditions),
		),
	}, nil
}

func main() {
	toolUseBehavior := flag.String(
		"t",
		"default",
		"The behavior to use for tool use.\n"+
			"  - 'default' will cause tool outputs to be sent to the model\n"+
			"  - 'first_tool' will cause the first tool result to be used as the final output\n"+
			"  - 'custom' will use a custom tool use behavior function")
	flag.Parse()

	var behavior agents.ToolUseBehavior
	var toolChoice modelsettings.ToolChoice

	switch *toolUseBehavior {
	case "default":
		behavior = agents.RunLLMAgain()
		toolChoice = nil
	case "first_tool":
		behavior = agents.StopOnFirstTool()
		toolChoice = modelsettings.ToolChoiceRequired
	case "custom":
		behavior = agents.ToolsToFinalOutputFunction(CustomToolUseBehavior)
		toolChoice = modelsettings.ToolChoiceRequired
	default:
		fmt.Printf("Error: invalid tool behavior: %q\n", *toolUseBehavior)
		syscall.Exit(1)
	}

	agent := agents.New("Weather agent").
		WithInstructions("You are a helpful agent.").
		WithTools(GetWeatherTool).
		WithToolUseBehavior(behavior).
		WithModelSettings(modelsettings.ModelSettings{
			ToolChoice: toolChoice,
		}).
		WithModel("gpt-4.1-nano")

	result, err := agents.Run(context.Background(), agent, "What's the weather in Tokyo?")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
}
