# LangSmith Tracing Integration for OpenAI Agents Go

This example demonstrates how to integrate LangSmith tracing with the OpenAI Agents Go SDK. LangSmith provides powerful tracing, monitoring, and debugging capabilities for AI applications.

## Features

- **Complete trace integration**: Captures all agent interactions, function calls, and LLM generations
- **Hierarchical span tracking**: Maintains proper parent-child relationships between operations
- **Metadata support**: Attach custom metadata and tags to traces
- **Concurrent processing**: Works alongside the default OpenAI backend processor
- **Error handling**: Robust error handling with graceful degradation

## Setup

### 1. Install Dependencies

This example uses the standard Go OpenAI Agents SDK - no additional dependencies required.

### 2. Environment Variables

Set up your LangSmith credentials:

```bash
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="your-project-name"  # Optional, defaults to "default"
```

Get your API key from [LangSmith Settings](https://smith.langchain.com/settings).

### 3. Run the Example

```bash
cd examples/langsmith_tracing
go run .
```

## Usage

### Basic Integration

```go
// Create the LangSmith processor
langsmithProcessor := TracingProcessor(LangSmithProcessorParams{
    ProjectName: "my-project",
    Metadata: map[string]any{
        "environment": "production",
        "version":     "1.0.0",
    },
    Tags: []string{"production", "golang"},
})

// Add to tracing system (works alongside default OpenAI processor)
tracing.AddTraceProcessor(langsmithProcessor)

// Or replace default processors entirely
// tracing.SetTraceProcessors([]tracing.Processor{langsmithProcessor})
```

### Configuration Options

```go
params := ProcessorParams{
    APIKey:      "your-api-key",      // Or use LANGSMITH_API_KEY env var
    APIURL:      "https://api.smith.langchain.com", // Default
    ProjectName: "my-project",        // Or use LANGSMITH_PROJECT env var  
    Metadata: map[string]any{         // Optional metadata for all traces
        "user_id": "123",
        "environment": "prod",
    },
    Tags: []string{"tag1", "tag2"},   // Optional tags for all traces
    Name: "Custom Workflow Name",     // Optional custom trace name
    HTTPClient: customHTTPClient,     // Optional custom HTTP client
}
```

## What Gets Traced

The LangSmith processor automatically captures:

- **Traces**: Complete workflow executions with metadata
- **Agent Spans**: Agent execution with tools and handoffs information
- **Function Spans**: Tool calls with inputs and outputs
- **Generation Spans**: LLM calls with model info, usage statistics, and configurations
- **Custom Spans**: Any custom spans you create

## Data Mapping

| OpenAI Agents Concept | LangSmith Concept | Notes |
|----------------------|-------------------|-------|
| Trace | Run (chain type) | Root-level workflow |
| Agent Span | Run (chain type) | Agent execution |
| Function Span | Run (tool type) | Tool/function calls |
| Generation Span | Run (llm type) | LLM generation |
| Span hierarchy | Parent/child runs | Maintained via dotted_order |

## Viewing Traces

After running your application:

1. Go to [LangSmith](https://smith.langchain.com/)
2. Navigate to your project
3. View traces in the "Tracing" section
4. Explore the hierarchical span structure
5. Analyze performance, costs, and debug issues

## Advanced Usage

### Multiple Processors

You can run multiple tracing processors simultaneously:

```go
// Send to both LangSmith and OpenAI backend
tracing.AddTraceProcessor(langsmithProcessor)
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
        },
    },
    func(ctx context.Context, trace tracing.Trace) error {
        // Your workflow logic
        return nil
    },
)
```

### Error Handling

The processor includes robust error handling:

- Graceful degradation when API key is missing
- HTTP retry logic for transient failures
- Detailed error logging for debugging
- Non-blocking operation (won't break your main workflow)

## Troubleshooting

### Common Issues

1. **Missing API Key**: Set `LANGSMITH_API_KEY` environment variable
2. **Network Issues**: Check firewall settings for `api.smith.langchain.com`
3. **Project Not Found**: Ensure project exists in LangSmith UI
4. **Rate Limiting**: Implement backoff in custom HTTP client if needed

### Debug Mode

Enable debug logging to see what's being sent to LangSmith:

```go
// Add debug logging (implement based on your logging preferences)
fmt.Printf("Sending trace data: %+v\n", traceData)
```

## Integration with Other Tools

This processor can work alongside other tracing tools:

- **OpenAI Backend**: Default processor for OpenAI's tracing dashboard
- **Custom Processors**: Your own logging, metrics, or monitoring systems
- **OpenTelemetry**: Can be adapted to send data to OTEL collectors

## Performance Considerations

- HTTP requests are non-blocking and won't slow down your main workflow
- Consider implementing batching for high-throughput applications
- Use custom HTTP client with appropriate timeouts and retry logic
- Monitor API usage in LangSmith dashboard

## License

Same as the OpenAI Agents Go SDK - Apache License 2.0.