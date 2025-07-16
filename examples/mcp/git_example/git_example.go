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
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

func main() {
	if _, err := exec.LookPath("uvx"); err != nil {
		fmt.Println("uvx is not installed. Please install it with `pip install uvx`.")
		os.Exit(1)
	}

	fmt.Print("Please enter the path to the git repository: ")
	_ = os.Stdout.Sync()
	line, _, err := bufio.NewReader(os.Stdin).ReadLine()
	if err != nil {
		panic(err)
	}
	directoryPath := string(line)
	fmt.Println(directoryPath)

	server := agents.NewMCPServerStdio(agents.MCPServerStdioParams{
		CacheToolsList: true, // Cache the tools list, for demonstration
		Command:        exec.Command("uvx", "mcp-server-git"),
	})

	err = server.Run(context.Background(), func(ctx context.Context, server *agents.MCPServerWithClientSession) error {
		return tracing.RunTrace(
			ctx, tracing.TraceParams{WorkflowName: "MCP Git Example"},
			func(ctx context.Context, _ tracing.Trace) error {
				return run(ctx, server, directoryPath)
			},
		)
	})
	if err != nil {
		panic(err)
	}
}

func run(ctx context.Context, mcpServer agents.MCPServer, directoryPath string) error {
	agent := agents.New("Assistant").
		WithInstructions(fmt.Sprintf("Answer questions about the git repository at %s, use that for repo_path.", directoryPath)).
		AddMCPServer(mcpServer).
		WithModel("gpt-4o")

	message := "Who's the most frequent contributor?"
	fmt.Println("\n-----\nRunning:", message)
	result, err := agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)

	message = "Summarize the last change in the repository."
	fmt.Println("\n-----\nRunning:", message)
	result, err = agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)

	return nil
}
