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
)

func CountAllOccurrences(s, substr string) int {
	fmt.Printf("🔧 CountAllOccurrences(%s, %s)\n", s, substr)

	count := 0
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			count++
		}
	}
	return count
}

func main() {

	// Requirements for agents.NewFunctionToolAny: Binary must contain DWARF debug info (default for `go build`).
	// Returns an error if debug info is stripped (e.g., with `-ldflags="-w"`).
	// Do not run with `go run`, as it strips debug information.
	countSubstringTool, err := agents.NewFunctionToolAny("", "", CountAllOccurrences)
	if err != nil {
		panic(err)
	}

	agent := agents.New("Assistant").
		WithInstructions("You're a helpful agent that can now count all occurrences of a substring in a string using the provided tool.").
		WithTools(countSubstringTool).
		WithModel("gpt-5-chat-latest")

	// Try using the tool first to see how it handles the request.
	// Then, test the agent's behavior without the tool to compare results.
	result, err := agents.Run(context.Background(), agent, "How many r's in Strawberrry?") // extra "r" intended

	if err != nil {
		panic(err)
	}

	fmt.Println(result.FinalOutput)
}
