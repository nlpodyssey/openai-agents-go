# OpenAI Agents Go SDK

The OpenAI Agents SDK is a lightweight yet powerful framework for building
multi-agent workflows. It is provider-agnostic, supporting the OpenAI Responses
and Chat Completions APIs, as well as other LLMs.

This is a Go port of [OpenAI Agents Python SDK](https://openai.github.io/openai-agents-python/)
(see its license [here](https://github.com/openai/openai-agents-python/tree/main?tab=MIT-1-ov-file#readme)).
This project aims at being as close as possible to the original Python
implementation, for both its behavior and the API. 

#### CAUTION: Work in Progress

A significant set of core functionalities has been already implemented and
works well. The project is under active development: we are implementing
missing features, adding tests, writing documentation, and refactoring the
code for a smoother usage. Please expect the SDK's API to change also
significantly before we release a first stable version.

### Core concepts:

1. **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs
2. **Handoffs**: A specialized tool call used by the Agents SDK for transferring control between agents
3. **Guardrails**: Configurable safety checks for input and output validation

Explore the [examples](examples) directory to see the SDK in action.

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
	"github.com/nlpodyssey/openai-agents-go/types/optional"
)

func main() {
	agent := &agents.Agent{
		Name:         "Assistant",
		Instructions: agents.StringInstructions("You are a helpful assistant"),
		Model:        optional.Value(agents.NewAgentModelName("gpt-4o")),
	}

	result, err := agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("Write a haiku about recursion in programming."),
	})
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
	"github.com/nlpodyssey/openai-agents-go/types/optional"
)

func main() {
	model := optional.Value(agents.NewAgentModelName("gpt-4o"))

	spanishAgent := &agents.Agent{
		Name:         "Spanish agent",
		Instructions: agents.StringInstructions("You only speak Spanish."),
		Model:        model,
	}

	englishAgent := &agents.Agent{
		Name:         "English agent",
		Instructions: agents.StringInstructions("You only speak English."),
		Model:        model,
	}

	triageAgent := &agents.Agent{
		Name: "Triage agent",
		Instructions: agents.StringInstructions(
			"Handoff to the appropriate agent based on the language of the request.",
		),
		Handoffs: []agents.AgentHandoff{spanishAgent, englishAgent},
		Model:    model,
	}

	result, err := agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: triageAgent,
		Input:         agents.InputString("Hola, ¿cómo estás?"),
	})
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
	"encoding/json"
	"fmt"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/nlpodyssey/openai-agents-go/types/optional"
)

type getWeatherParams struct {
	City string `json:"city"`
}

func getWeather(params getWeatherParams) string {
	return fmt.Sprintf("The weather in %s is sunny.", params.City)
}

var (
	getWeatherParamsJSONSchema = map[string]any{
		"type":                 "object",
		"required":             []string{"city"},
		"additionalProperties": false,
		"properties": map[string]any{
			"city": map[string]any{
				"type": "string",
			},
		},
	}
	getWeatherTool = agents.FunctionTool{
		Name:             "GetWeather",
		ParamsJSONSchema: getWeatherParamsJSONSchema,
		OnInvokeTool: func(_ context.Context, _ *runcontext.RunContextWrapper, args string) (any, error) {
			var params getWeatherParams
			err := json.Unmarshal([]byte(args), &params)
			if err != nil {
				return nil, err
			}
			return getWeather(params), nil
		},
	}
	agent = &agents.Agent{
		Name:         "Hello world",
		Instructions: agents.StringInstructions("You are a helpful agent."),
		Tools:        []agents.Tool{getWeatherTool},
		Model:        optional.Value(agents.NewAgentModelName("gpt-4o")),
	}
)

func main() {
	result, err := agents.Runner().Run(context.Background(), agents.RunParams{
		StartingAgent: agent,
		Input:         agents.InputString("What's the weather in Tokyo?"),
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.FinalOutput)
	// The weather in Tokyo is sunny.
}
```

## The agent loop

When you call `agents.Runner().Run()`, we run a loop until we get a final output.

1. We call the LLM, using the model and settings on the agent, and the message history.
2. The LLM returns a response, which may include tool calls.
3. If the response has a final output (see below for more on this), we return it and end the loop.
4. If the response has a handoff, we set the agent to the new agent and go back to step 1.
5. We process the tool calls (if any) and append the tool responses messages. Then we go to step 1.

There is a `MaxTurns` parameter that you can use to limit the number of times the loop executes.

### Final output

Final output is the last thing the agent produces in the loop.

1.  If you set an `OutputSchema` on the agent, the final output is when the  LLM returns something of that type. We use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) for this.
2.  If there's no `OutputSchema` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

As a result, the mental model for the agent loop is:

1. If the current agent has an `OutputSchema`, the loop runs until the agent produces structured output matching that type.
2. If the current agent does not have an `OutputSchema`, the loop runs until the current agent produces a message without any tool calls/handoffs.

## Common agent patterns

The Agents SDK is designed to be highly flexible, allowing you to model a wide
range of LLM workflows including deterministic flows, iterative loops, and more.
See examples in [`examples/agent_patterns`](examples/agent_patterns).

## Acknowledgments

We would like to thank the OpenAI team for creating the original [OpenAI Agents Python SDK](https://openai.github.io/openai-agents-python/) and the [official OpenAI Go client library](https://github.com/openai/openai-go), which serve as the foundation for this Go implementation.
