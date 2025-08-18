# Custom LLM providers (Go Examples)

This directory mirrors the official Python examples for using non-OpenAI providers with the OpenAI Agents SDK, adapted for Go.

## Examples

- custom_example_provider: Provide a custom `ModelProvider` and use it via `Runner`.
- custom_example_agent: Set a custom client per agent using `WithModelInstance(...)`.
- custom_example_global: Set a global default OpenAI client and default API (`chat_completions`).
- custom_example_litellm: Use LiteLLM (Docker) as an OpenAI-compatible proxy. Includes MultiProvider with prefix routing and a global-default example.

## Prerequisites

For the first three examples, set the following environment variables (or edit the code):

```bash
export EXAMPLE_BASE_URL="http://your-provider-host"
export EXAMPLE_API_KEY="your-provider-api-key"
export EXAMPLE_MODEL_NAME="model-name"
```

For LiteLLM, start a Docker container (requires `OPENAI_API_KEY` for upstream providers):

```bash
docker run \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -p 4000:4000 \
  ghcr.io/berriai/litellm:main-stable \
  --config /app/config.yaml \
  --detailed_debug
```

Then run:

```bash
go run ./examples/model_providers/custom_example_provider
go run ./examples/model_providers/custom_example_agent
go run ./examples/model_providers/custom_example_global
go run ./examples/model_providers/custom_example_litellm
```

## Notes on Parity with Python

The Go examples match the Python variants conceptually:

- Per-agent client: `custom_example_agent.go` ↔ `custom_example_agent.py`
- Global default client: `custom_example_global.go` ↔ `custom_example_global.py`
- Runner-level provider: `custom_example_provider.go` ↔ `custom_example_provider.py`
- LiteLLM routing: `custom_example_litellm/main.go` shows direct provider, MultiProvider prefix routing (`litellm/...`), and global default — corresponding to Python's `litellm_auto.py` and `litellm_provider.py` patterns.


