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

func main() {
	agent := agents.New("MCP Tool Example").
		WithInstructions("You are a helpful agent that can use remote MCP servers to perform tasks.").
		WithTools(agents.MCPTool{
			ServerLabel:     "deepwiki",
			ServerURL:       "https://mcp.deepwiki.com/mcp",
			RequireApproval: "never",
			AllowedTools:    []string{"ask_question"},
		}).
		WithModel("gpt-4.1")

	result, err := agents.Run(
		context.Background(),
		agent,
		"What transport protocols are supported in the 2025-03-26 version of the MCP spec?",
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(result.FinalOutput)
}
