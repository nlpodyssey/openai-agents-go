# LiteLLM Docker Integration Example

This example demonstrates how to integrate the OpenAI Agents Go SDK with LiteLLM running locally in Docker. LiteLLM acts as a unified proxy that provides OpenAI-compatible APIs for multiple LLM providers including OpenAI, Anthropic, Google, and local models.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Go Application │────│  LiteLLM Docker │────│  LLM Providers  │
│  (Agents SDK)   │    │  (Port 4000)    │    │  (OpenAI, etc.) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Docker** installed and running
2. **OpenAI API Key** (set as environment variable)
3. **Go 1.21+** for running the example

### Step 1: Start LiteLLM Docker Container

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-openai-api-key-here"

# Start LiteLLM with the provided config
docker run \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -p 4000:4000 \
  ghcr.io/berriai/litellm:main-stable \
  --config /app/config.yaml \
  --detailed_debug
```

### Step 2: Verify LiteLLM is Running

```bash
# Test the endpoint
curl --location 'http://localhost:4000/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "openai-gpt-4o",
    "messages": [{"role": "user", "content": "Hello from LiteLLM!"}],
    "max_tokens": 50
  }'
```

### Step 3: Run the Go Example

```bash
# From the example directory
go run main.go
```

## 📋 Example Features

The example demonstrates **three different integration patterns**:

### 1. Direct LiteLLM Provider
```go
// Create a dedicated LiteLLM provider
litellmProvider := NewLiteLLMProvider("", "")

// Use it directly with an agent
result, err := (agents.Runner{
    Config: agents.RunConfig{
        ModelProvider: litellmProvider,
        Model: param.NewOpt(agents.NewAgentModelName("openai-gpt-4o")),
    },
}).Run(context.Background(), agent, "Your prompt here")
```

### 2. MultiProvider with Prefix Routing
```go
// Setup multiple providers with prefixes
providerMap := agents.NewMultiProviderMap()
providerMap.AddProvider("litellm", NewLiteLLMProvider("", ""))

multiProvider := agents.NewMultiProvider(agents.NewMultiProviderParams{
    ProviderMap: providerMap,
})

// Use with prefix-based model names
agent.WithModel("litellm/openai-gpt-4o")
```

### 3. Global Default Provider
```go
// Set LiteLLM as the global default
litellmClient := agents.NewOpenaiClient(
    param.NewOpt("http://localhost:4000"),
    param.NewOpt("dummy-key"),
)
agents.SetDefaultOpenaiClient(litellmClient, false)

// Now all agents use LiteLLM by default
agent.WithModel("openai-gpt-4o") // No prefix needed
```

## 🔧 Configuration

### LiteLLM Config (`litellm_config.yaml`)

The configuration supports multiple providers:

```yaml
model_list:
  # OpenAI Models
  - model_name: openai-gpt-4o
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  # Anthropic Models
  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20240620
      api_key: os.environ/ANTHROPIC_API_KEY

  # Local Ollama Models
  - model_name: ollama-llama3
    litellm_params:
      model: ollama/llama3
      api_base: http://localhost:11434
```

### Environment Variables

Set these environment variables for different providers:

```bash
# Required for OpenAI models
export OPENAI_API_KEY="sk-..."

# Optional for Anthropic models
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional for Google models
export GOOGLE_API_KEY="..."
```

## 🎯 Advanced Features

### Model Fallbacks

LiteLLM supports automatic fallbacks if a model is unavailable:

```yaml
general_settings:
  fallbacks:
    - openai-gpt-4o
    - openai-gpt-4o-mini
```

### Custom Headers and Settings

```go
// Add custom headers for specific providers
modelSettings := modelsettings.ModelSettings{
    ExtraHeaders: map[string]string{
        "X-LiteLLM-Provider": "anthropic",
        "X-Custom-Header":    "value",
    },
}

agent.WithModelSettings(modelSettings)
```

### Health Check

The example includes a connection check:

```go
func checkLiteLLMConnection() bool {
    // Attempts a simple request to verify LiteLLM is accessible
    // Returns true if LiteLLM is running and responsive
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: connection refused on port 4000
   ```
   **Solution**: Ensure Docker container is running and port 4000 is not blocked.

2. **API Key Errors**
   ```
   Error: 401 Unauthorized
   ```
   **Solution**: Check that your API keys are properly set in environment variables.

3. **Model Not Found**
   ```
   Error: model "xyz" not found
   ```
   **Solution**: Verify the model name matches what's defined in `litellm_config.yaml`.

### Debug Mode

Enable verbose logging in LiteLLM:

```yaml
general_settings:
  set_verbose: true
```

Or check Docker logs:

```bash
docker logs <container-id>
```

## 🌐 Multi-Provider Setup

### Adding More Providers

To add support for additional providers, update `litellm_config.yaml`:

```yaml
model_list:
  # Hugging Face
  - model_name: hf-codellama
    litellm_params:
      model: huggingface/codellama/CodeLlama-7b-Instruct-hf
      api_key: os.environ/HUGGINGFACE_API_KEY

  # Cohere
  - model_name: cohere-command
    litellm_params:
      model: cohere/command-r-plus
      api_key: os.environ/COHERE_API_KEY
```

## 🎉 Benefits

1. **Provider Flexibility**: Switch between providers without code changes
2. **Cost Optimization**: Route to cheaper models when appropriate
3. **High Availability**: Automatic fallbacks if a provider is down
4. **Local Development**: Test with local models via Ollama
5. **Unified Interface**: Single API for all providers

## 📚 Further Reading

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [OpenAI Agents Go SDK](https://github.com/nlpodyssey/openai-agents-go)
- [Supported LiteLLM Providers](https://docs.litellm.ai/docs/providers)

---

This example showcases the power and flexibility of the OpenAI Agents Go SDK's provider system, enabling seamless integration with any OpenAI-compatible service!
