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
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

func main() {
	agent := agents.New("Assistant").WithModel("gpt-4o")

	input := agents.InputList(
		agents.SystemMessage("You are a helpful assistant."),
		agents.DeveloperMessage(fmt.Sprintf("Current time is %s", time.Now().Format(time.Kitchen))),
		"What's the time?", // strings become user messages
	)

	result, err := agents.RunInputs(context.Background(), agent, input)
	if err != nil {
		panic(err)
	}

	fmt.Println(result.FinalOutput)
	// The current time is 4:09 PM.
}
