# Parallel Search MCP

Connect a Go agent to [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
using the SDK's Streamable HTTP transport. The remote server offers `web_search`
and `web_fetch` without a Parallel API key. The anonymous free tier has rate
limits and is intended for exploration and light use.

From the repository root, run:

```sh
export OPENAI_API_KEY=your-openai-key
go run ./examples/mcp/parallel_search -query 'What is the OpenAI Agents Go SDK?'
```

This uses `gpt-4o` to choose tools and write an answer with source URLs. Model
inference is billed separately by OpenAI. No Parallel credentials are read or sent.

To try both tools without a model or any API keys:

```sh
go run ./examples/mcp/parallel_search -direct \
  -query 'OpenAI Agents Go SDK MCP support' \
  -url https://github.com/nlpodyssey/openai-agents-go
```

Direct mode discovers the remote tools, converts them to SDK function tools, and
prints the search results and fetched page content. It fetches the supplied URL;
it does not automatically choose a search result. Both calls share a fresh
conversation ID. The agent receives the same ID in its instructions.

The example sets `User-Agent: openai-agents-go/parallel-search-example` on MCP
requests, uses a 60-second HTTP timeout and a three-minute overall deadline, and
supports cancellation with Ctrl+C. Tracing is disabled for this example.

Run the offline transport and agent tests with:

```sh
go test ./examples/mcp/parallel_search
```
