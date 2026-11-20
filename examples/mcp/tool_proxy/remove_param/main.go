package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/openai/openai-go/v3/packages/param"
)

const mcpPort = 9010

func getRunner() agents.Runner {
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		apiKey = "ollama"
	}

	provider := agents.NewOpenAIProvider(
		agents.OpenAIProviderParams{
			APIKey:  param.NewOpt(apiKey),
			BaseURL: param.NewOpt(baseURL),
		},
	)

	runner := agents.Runner{
		Config: agents.RunConfig{
			ModelProvider:   provider,
			TracingDisabled: true,
		},
	}

	return runner
}

func getMCP() agents.MCPServer {
	server := agents.NewMCPServerStreamableHTTP(agents.MCPServerStreamableHTTPParams{
		Name: "Complex MCP Server",
		URL:  "http://localhost:" + strconv.Itoa(mcpPort),
		ToolProxyOrchestrator: agents.MCPToolProxyOrchestrator{Proxies: []agents.MCPToolProxy{
			{
				Conditionals: &agents.MCPToolProxyExactNameConditionals{Names: []string{"get_secret_code"}},
				Processors: []agents.MCPToolProxyProcessor{
					&agents.MCPToolProxyRemoveInputParamsProcessor{
						Params:                  []string{"grade"},
						RemoveRequiredBehaviour: agents.MCPToolProxyRemoveRequiredBehaviourRemove,
					},
				},
			},
		}},
	})
	return server
}

func main() {
	go runServer(mcpPort)
	time.Sleep(2 * time.Second) // time to start MCP server
	mcpServer := getMCP()

	err := mcpServer.Connect(context.Background())
	if err != nil {
		panic(err)
	}
	defer func(mcpServer agents.MCPServer, ctx context.Context) {
		err := mcpServer.Cleanup(ctx)
		if err != nil {
			panic(err)
		}
	}(mcpServer, context.Background())

	agent := agents.New("Agent").
		WithInstructions("Return secret code of user").
		WithModel("qwen2.5:3b"). // you can change for your model. ollama may be useful for testing
		AddMCPServer(mcpServer)

	runner := getRunner()
	result, err := runner.Run(context.Background(), agent, "Give me secret code of Ivan Ivanov (grade junior) plz")
	if err != nil {
		panic(err)
	}

	fmt.Println(result.FinalOutput)
}
