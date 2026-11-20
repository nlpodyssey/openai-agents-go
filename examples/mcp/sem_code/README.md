# Local code navigation with Sem

This optional example connects to [Sem](https://github.com/Ataraxy-Labs/sem)
over stdio, exposing only `sem_find` and `sem_grep` to the agent. Install Sem
using its upstream instructions and confirm `sem --version` works. Tested with
Sem 0.27.0. No changes to the SDK or default tools are needed.

From the SDK repository root, look up an existing symbol without model credentials:

```sh
go run ./examples/mcp/sem_code -direct -repo /absolute/path/to/checkout -query parse_config
```

Replace both the path and symbol. Direct mode uses the SDK's MCP-to-function-tool
adapter, requests a 1,200-token context budget, and prints the result. An empty
lookup is not evidence that a symbol or relationship does not exist.

For a model-driven question, configure `OPENAI_API_KEY` and omit `-direct`:

```sh
go run ./examples/mcp/sem_code -repo /absolute/path/to/checkout -query 'Find parse_config and explain how its callers use the result.'
```

Use `-sem /absolute/path/to/sem` or `-model MODEL` if needed. Tracing is disabled
in both modes; agent mode still sends the question and retrieved source to the
configured model provider. Connections are cleaned up by `server.Run`, and a
three-minute context deadline bounds the example.

Sem may write a local index, including during lookup. The full server exposes
more capabilities, including command-running validation; the example filters
model-visible tools but does not sandbox the server. `SEM_REPO` selects a checkout,
not an access-control boundary. Use trusted repositories and host permissions.

```sh
go test ./examples/mcp/sem_code
```
