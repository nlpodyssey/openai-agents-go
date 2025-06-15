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

1.  If you set an `OutputSchema` on the agent, the final output is when the  LLM returns something of that type. We use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) for this.
2.  If there's no `OutputSchema` (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.

As a result, the mental model for the agent loop is:

1. If the current agent has an `OutputSchema`, the loop runs until the agent produces structured output matching that type.
2. If the current agent does not have an `OutputSchema`, the loop runs until the current agent produces a message without any tool calls/handoffs.

## Common agent patterns

The Agents SDK is designed to be highly flexible, allowing you to model a wide
range of LLM workflows including deterministic flows, iterative loops, and more.
See examples in [`examples/agent_patterns`](examples/agent_patterns).

## Authors

This project is currently being developed by [Matteo Grella](https://github.com/matteo-grella) and [Marco Nicola](https://github.com/marco-nicola) as an early-stage port of [OpenAI's Agents SDK](https://openai.github.io/openai-agents-python/), aimed at supporting its initial adoption by Go developers and offering something potentially useful to the OpenAI team.

## Acknowledgments

We would like to thank the OpenAI team for creating the original [OpenAI Agents Python SDK](https://openai.github.io/openai-agents-python/) and the [official OpenAI Go client library](https://github.com/openai/openai-go), which serve as the foundation for this Go implementation.
