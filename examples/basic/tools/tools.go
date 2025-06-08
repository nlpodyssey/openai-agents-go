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

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/openai/openai-go/packages/param"
)

type Weather struct {
	City             string `json:"city"`
	TemperatureRange string `json:"temperature_range"`
	Conditions       string `json:"conditions"`
}

type GetWeatherArgs struct {
	City string `json:"city"`
}

func GetWeather(args GetWeatherArgs) Weather {
	fmt.Println("[debug] GetWeather called")
	return Weather{
		City:             args.City,
		TemperatureRange: "14-20C",
		Conditions:       "Sunny with wind.",
	}
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
	OnInvokeTool: func(_ context.Context, _ *runcontext.Wrapper, arguments string) (any, error) {
		var args GetWeatherArgs
		err := json.Unmarshal([]byte(arguments), &args)
		if err != nil {
			return nil, err
		}
		w := GetWeather(args)
		out, err := json.Marshal(w)
		if err != nil {
			return nil, err
		}
		return string(out), nil
	},
	StrictJSONSchema: param.NewOpt(true),
}

func main() {
	agent := &agents.Agent{
		Name:         "Hello world",
		Instructions: agents.StringInstructions("You are a helpful agent."),
		Model:        param.NewOpt(agents.NewAgentModelName("gpt-4.1-nano")),
		Tools: []agents.Tool{
			GetWeatherTool,
		},
	}

	ctx := context.Background()

	result, err := agents.Runner().Run(ctx, agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("What's the weather in Tokyo?"),
	})

	if err != nil {
		panic(err)
	}

	fmt.Println(result.FinalOutput)
}
