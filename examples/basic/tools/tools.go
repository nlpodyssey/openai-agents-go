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

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tools"
)

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
		Conditions:       "Sunny with wind.",
	}, nil
}

var GetWeatherTool = tools.NewFunctionTool("get_weather", "", GetWeather)

func main() {
	agent := agents.New("Hello world").
		WithInstructions("You are a helpful agent.").
		WithModel("gpt-4.1-nano").
		WithTools(GetWeatherTool)

	ctx := context.Background()

	result, err := agents.Run(ctx, agent, "What's the weather in Tokyo?")

	if err != nil {
		panic(err)
	}

	fmt.Println(result.FinalOutput)
}
