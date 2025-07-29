# Traceloop Integration for OpenAI Agents Go

This integration allows you to send traces from the OpenAI Agents Go SDK to [Traceloop](https://www.traceloop.com/openllmetry), an open-source observability platform for LLM applications based on OpenTelemetry.

## Features

- **Complete trace integration**: Captures all agent interactions, function calls, and LLM generations
- **Workflow and task organization**: Maps traces to Traceloop workflows and spans to tasks
- **LLM call tracking**: Detailed prompt and completion logging with usage statistics
- **OpenTelemetry based**: Built on OpenTelemetry standards for maximum compatibility
- **Multiple destinations**: Send to Traceloop, Datadog, Honeycomb, and [20+ other platforms](https://github.com/traceloop/go-openllmetry#-supported-and-tested-destinations)

## Setup

### 1. Install Dependencies

First, install the Traceloop SDK:

```bash
go get github.com/traceloop/go-openllmetry/traceloop-sdk
```

### 2. Get API Key

Sign up at [Traceloop](https://app.traceloop.com) to get your API key.

### 3. Environment Variables

Set your Traceloop API key:

```bash
export TRACELOOP_API_KEY="your-traceloop-api-key"
```

### 4. Run the Example

```bash
cd examples/tracing/traceloop
go run main.go
```

## Usage

### Basic Integration

```go
package main

import (
    "context"
    "os"
    
    "github.com/nlpodyssey/openai-agents-go/agents"
    "github.com/nlpodyssey/openai-agents-go/tracing"
    "github.com/nlpodyssey/openai-agents-go/tracing/wrappers/traceloop"
)

func main() {
    ctx := context.Background()
    
    // Create the Traceloop processor
    traceloopProcessor, err := traceloop.NewTracingProcessor(ctx, traceloop.ProcessorParams{
        APIKey:  os.Getenv("TRACELOOP_API_KEY"),
        BaseURL: "api.traceloop.com",
        Metadata: map[string]any{
            "environment": "production",
            "version":     "1.0.0",
        },
        Tags: []string{"production", "golang"},
    })
    if err != nil {
        panic(err)
    }
    
    // Add to tracing system (works alongside default OpenAI processor)
    tracing.AddTraceProcessor(traceloopProcessor)
    
    // Or replace default processors entirely
    // tracing.SetTraceProcessors([]tracing.Processor{traceloopProcessor})
    
    // Use agents normally - tracing happens automatically
    agent := agents.New("Assistant").WithModel("gpt-4o-mini")
    result, err := agents.Run(ctx, agent, "Hello!")
    // ... rest of your code
}
```

### Configuration Options

```go
params := traceloop.ProcessorParams{
    APIKey:  "your-api-key",                    // Required
    BaseURL: "api.traceloop.com",               // Default, use "api-staging.traceloop.com" for staging
    Metadata: map[string]any{                   // Optional metadata for all workflows
        "user_id": "123",
        "environment": "prod",
    },
    Tags: []string{"tag1", "tag2"},            // Optional tags for all workflows
}
```

## What Gets Traced

The Traceloop processor automatically captures:

- **Workflows**: Complete agent execution flows with metadata
- **Tasks**: Individual agent spans, function calls, and LLM generations
- **LLM Calls**: Full prompt and completion data with usage statistics
- **Agent Handoffs**: Transitions between different agents
- **Tool Usage**: Function calls with inputs and outputs
- **Error Tracking**: Span errors and failures

## Data Mapping

| OpenAI Agents Concept | Traceloop Concept | Notes |
|----------------------|-------------------|-------|
| Trace | Workflow | Root-level workflow execution |
| Agent Span | Task | Agent execution task |
| Function Span | Task | Tool/function call task |
| Generation/Response Span | Task + LLM Span | LLM call with prompt/completion |
| Span hierarchy | Task nesting | Maintained within workflows |
| Metadata | Workflow attributes | User ID, session ID, etc. |

## Viewing Traces

After running your application:

1. Go to [Traceloop Dashboard](https://app.traceloop.com)
2. Navigate to your project
3. View workflows in the "Traces" section
4. Explore task hierarchy and LLM interactions
5. Analyze performance, costs, and debug issues

## Advanced Usage

### Multiple Processors

You can run multiple tracing processors simultaneously:

```go
// Send to both Traceloop and OpenAI backend
tracing.AddTraceProcessor(traceloopProcessor)
tracing.AddTraceProcessor(customProcessor)
```

### Custom Metadata Per Trace

```go
err := tracing.RunTrace(
    ctx,
    tracing.TraceParams{
        WorkflowName: "Custom Workflow",
        GroupID:      "conversation-123",  // Links related traces
        Metadata: map[string]any{
            "user_id": "abc",
            "session_id": "xyz",
            "custom_field": "value",
        },
    },
    func(ctx context.Context, trace tracing.Trace) error {
        // Your workflow logic
        return nil
    },
)
```

### Alternative Destinations

Since Traceloop is built on OpenTelemetry, you can easily send data to other platforms. See the [Traceloop documentation](https://github.com/traceloop/go-openllmetry#-supported-and-tested-destinations) for supported destinations including:

- Datadog
- Honeycomb  
- New Relic
- Grafana
- Langfuse
- And many more...

## Troubleshooting

### Common Issues

1. **Missing API Key**: Set `TRACELOOP_API_KEY` environment variable
2. **Import Errors**: Run `go get github.com/traceloop/go-openllmetry/traceloop-sdk`
3. **Network Issues**: Check firewall settings for `api.traceloop.com`
4. **Empty Traces**: Ensure agents are making LLM calls (not just returning cached responses)

### Debug Mode

Enable debug logging to see what's being sent to Traceloop:

```go
// Add debug logging in your processor initialization
fmt.Printf("Traceloop processor created successfully\n")
```

## Comparison with LangSmith

| Feature | Traceloop | LangSmith |
|---------|-----------|-----------|
| **Base Technology** | OpenTelemetry | Custom API |
| **Destinations** | 20+ platforms | LangSmith only |
| **Open Source** | ✅ | ❌ |
| **Self-hosting** | ✅ | Limited |
| **Enterprise Features** | Available | Available |
| **Learning Curve** | Moderate | Easy |
| **Integration Effort** | Low | Low |

## Performance Considerations

- OpenTelemetry-based architecture is highly optimized
- Background processing with automatic batching
- Non-blocking operation (won't slow down your main workflow)
- Configurable sampling rates available
- Memory-efficient span tracking

## License

Same as the OpenAI Agents Go SDK - Apache License 2.0.

## Links

- [Traceloop Website](https://www.traceloop.com)
- [Traceloop Go SDK](https://github.com/traceloop/go-openllmetry)
- [OpenTelemetry Go](https://opentelemetry.io/docs/languages/go/)
- [Supported Destinations](https://github.com/traceloop/go-openllmetry#-supported-and-tested-destinations) 