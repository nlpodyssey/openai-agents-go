# OpenAI Agents Go SDK

The OpenAI Agents SDK is a lightweight yet powerful framework for building
multi-agent workflows. It is provider-agnostic, supporting the OpenAI Responses
and Chat Completions APIs, as well as other LLMs.

This is a Go port of [OpenAI Agents Python SDK](https://openai.github.io/openai-agents-python/)
(see its license [here](https://github.com/openai/openai-agents-python/tree/main?tab=MIT-1-ov-file#readme)).
This project aims at being as close as possible to the original Python
implementation, for both its behavior and the API. 

### Core concepts:

1. **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs
2. **Handoffs**: A specialized tool call used by the Agents SDK for transferring control between agents
3. **Guardrails**: Configurable safety checks for input and output validation

Explore the [examples](examples) directory to see the SDK in action:

| Directory | Description |
|-----------|-------------|
| [basic](examples/basic) | Core features such as hello world, streaming, prompt templates, and tools. |
| [agent_patterns](examples/agent_patterns) | Common agent design patterns including routing, guardrails, and parallelization. |
| [customer_service](examples/customer_service) | Multi-agent airline support scenario using handoffs and tools. |
| [financial_research_agent](examples/financial_research_agent) | Coordinated agents performing financial analysis and report writing. |
| [handoffs](examples/handoffs) | Techniques for filtering messages and handing off between agents. |
| [hosted_mcp](examples/hosted_mcp) | Hosted Model Context Protocol examples, including simple and approval flows. |
| [mcp](examples/mcp) | Running local MCP servers and clients for filesystems, git, prompts, and streaming. |
| [model_providers](examples/model_providers) | Integrating custom model providers and proxies like LiteLLM. |
| [research_bot](examples/research_bot) | General research bot combining planner, search, and writer agents. |
| [repl](examples/repl) | Command-line REPL for interactive experimentation. |
| [session](examples/session) | Demonstrates persistent session memory across multiple runs. |
| [tools](examples/tools) | Usage of built-in tools such as code interpreter, computer use, file search, and web search. |
| [voice](examples/voice) | Static and streaming voice response examples. |

## Installation

```
go get github.com/nlpodyssey/openai-agents-go
```

## Hello world example

```go
package main

import (
	"context"
	"fmt"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

func main() {
    agent := agents.New("Assistant").
            WithInstructions("You are a helpful assistant").
            WithModel("gpt-4o")

    result, err := agents.Run(context.Background(), agent, "Write a haiku about recursion in programming.")
    if err != nil {
        panic(err)
    }

    fmt.Println(result.FinalOutput)
    // Function calls itself,
    // Deep within the endless loop,
    // Code mirrors its form.
}
```

(_If running this, ensure you set the `OPENAI_API_KEY` environment variable_)

## Handoffs example

```go
package main

import (
	"context"
	"fmt"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

func main() {
    spanishAgent := agents.New("Spanish agent").
        WithInstructions("You only speak Spanish.").
        WithModel("gpt-4o")

    englishAgent := agents.New("English agent").
        WithInstructions("You only speak English.").
        WithModel("gpt-4o")

    triageAgent := agents.New("Triage agent").
        WithInstructions("Handoff to the appropriate agent based on the language of the request.").
        WithAgentHandoffs(spanishAgent, englishAgent).
        WithModel("gpt-4o")
	
    result, err := agents.Run(context.Background(), triageAgent, "Hola, ¿cómo estás?")
    if err != nil {
        panic(err)
    }

    fmt.Println(result.FinalOutput)
    // ¡Hola! Estoy bien, gracias. ¿Y tú cómo estás?
}
```

## Functions example

```go
package main

import (
        "context"
        "fmt"

        "github.com/nlpodyssey/openai-agents-go/agents"
)

// Tool params type
type GetWeatherParams struct {
        City string `json:"city"`
}

// Tool implementation
func getWeather(_ context.Context, params GetWeatherParams) (string, error) {
        return fmt.Sprintf("The weather in %s is sunny.", params.City), nil
}

// Tool registration (using SDK's NewFunctionTool)
var getWeatherTool = agents.NewFunctionTool("GetWeather", "", getWeather)

func main() {
        agent := agents.New("Hello world").
                WithInstructions("You are a helpful agent.").
                WithModel("gpt-4o").
                WithTools(getWeatherTool)
	
        result, err := agents.Run(context.Background(), agent, "What's the weather in Tokyo?")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
	// The weather in Tokyo is sunny.
}
```

## The agent loop

When you call `agents.Run()`, we run a loop until we get a final output.

1. We call the LLM, using the model and settings on the agent, and the message history.
2. The LLM returns a response, which may include tool calls.
3. If the response has a final output (see below for more on this), we return it and end the loop.
4. If the response has a handoff, we set the agent to the new agent and go back to step 1.
5. We process the tool calls (if any) and append the tool responses messages. Then we go to step 1.

There is a `MaxTurns` parameter that you can use to limit the number of times the loop executes.

### Final output

Final output is the last thing the agent produces in the loop.

1.  If you set an `OutputType` on the agent, the final output is when the  LLM returns something of that type. We use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) for this.
2.  If there's no `OutputType` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

As a result, the mental model for the agent loop is:

1. If the current agent has an `OutputType`, the loop runs until the agent produces structured output matching that type.
2. If the current agent does not have an `OutputType`, the loop runs until the current agent produces a message without any tool calls/handoffs.

## Common agent patterns

The Agents SDK is designed to be highly flexible, allowing you to model a wide
range of LLM workflows including deterministic flows, iterative loops, and more.
See examples in [`examples/agent_patterns`](examples/agent_patterns).

## Authors

This project was started by [Matteo Grella](https://github.com/matteo-grella) and [Marco Nicola](https://github.com/marco-nicola) as a port of [OpenAI's Agents SDK](https://openai.github.io/openai-agents-python/), aimed at supporting its adoption by Go developers and offering something potentially useful to the OpenAI team.  
It has since evolved with community contributions, and we welcome new ideas, improvements, and pull requests from anyone interested in shaping its future.

## Acknowledgments

We would like to thank the OpenAI team for creating the original [OpenAI Agents Python SDK](https://openai.github.io/openai-agents-python/) and the [official OpenAI Go client library](https://github.com/openai/openai-go), which serve as the foundation for this Go implementation.

We also acknowledge [Anthropic, PBC](https://www.anthropic.com) for creating and maintaining the [Model Context Protocol](https://github.com/modelcontextprotocol), a crucial dependency for the MCP functionality in this framework, and particularly the [MCP Go SDK](https://github.com/modelcontextprotocol/go-sdk).