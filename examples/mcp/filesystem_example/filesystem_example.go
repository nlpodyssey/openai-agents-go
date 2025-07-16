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
	"os"
	"os/exec"
	"path/filepath"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/tracing"
)

func main() {
	if _, err := exec.LookPath("npx"); err != nil {
		fmt.Println("npx is not installed. Please install it with `npm install -g npx`.")
		os.Exit(1)
	}

	currentDir, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	samplesDir := filepath.Join(currentDir, "sample_files")

	server := agents.NewMCPServerStdio(agents.MCPServerStdioParams{
		Name:    "Filesystem Server, via npx",
		Command: exec.Command("npx", "-y", "@modelcontextprotocol/server-filesystem", samplesDir),
	})

	err = server.Run(context.Background(), func(ctx context.Context, server *agents.MCPServerWithClientSession) error {
		traceID := tracing.GenTraceID()
		return tracing.RunTrace(
			ctx, tracing.TraceParams{WorkflowName: "MCP Filesystem Example", TraceID: traceID},
			func(ctx context.Context, _ tracing.Trace) error {
				fmt.Printf("View trace: https://platform.openai.com/traces/trace?trace_id=%s\n", traceID)
				return run(ctx, server)
			},
		)
	})
	if err != nil {
		panic(err)
	}
}

func run(ctx context.Context, mcpServer agents.MCPServer) error {
	agent := agents.New("Assistant").
		WithInstructions("Use the tools to read the filesystem and answer questions based on those files.").
		AddMCPServer(mcpServer).
		WithModel("gpt-4o")

	// List the files it can read
	message := "Read the files and list them."
	fmt.Println("Running:", message)
	result, err := agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)

	// Ask about books
	message = "What is my #1 favorite book?"
	fmt.Println("\nRunning:", message)
	result, err = agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)

	// Ask a question that reads then reasons.
	message = "Look at my favorite songs. Suggest one new song that I might like."
	fmt.Println("\nRunning:", message)
	result, err = agents.Run(ctx, agent, message)
	if err != nil {
		return err
	}
	fmt.Println(result.FinalOutput)

	return nil
}
