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
	"math/rand"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

type Style string

const (
	StyleHaiku  Style = "haiku"
	StylePirate Style = "pirate"
	StyleRobot  Style = "robot"
)

type customContextKey struct{}

type CustomContext struct {
	Style Style
}

func CustomInstructions(ctx context.Context, _ *agents.Agent) (string, error) {
	customContext := ctx.Value(customContextKey{}).(CustomContext)
	switch customContext.Style {
	case StyleHaiku:
		return "Only respond in haikus.", nil
	case StylePirate:
		return "Respond as a pirate.", nil
	case StyleRobot:
		return "Respond as a robot and say 'beep boop' a lot.", nil
	default:
		return "", fmt.Errorf("unexpected style %q", customContext.Style)
	}
}

var Agent = agents.NewAgent().
	WithName("Chat agent").
	WithInstructionsFunc(CustomInstructions).
	WithModel("gpt-4.1-nano")

func main() {
	choice := []Style{StyleHaiku, StylePirate, StyleRobot}[rand.Intn(3)]
	customContext := CustomContext{Style: choice}
	fmt.Printf("Using style: %s\n", choice)

	userMessage := "Tell me a joke."
	fmt.Printf("User: %s\n", userMessage)

	ctx := context.WithValue(context.Background(), customContextKey{}, customContext)

	result, err := agents.Runner().Run(ctx, agents.RunParams{
		StartingAgent: Agent,
		Input:         agents.InputString(userMessage),
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Assistant: %s\n", result.FinalOutput)
}
