# deed.page MCP

Connect a Go agent to the [deed.page](https://deed.page) remote MCP server
using the SDK's Streamable HTTP transport. The agent writes an HTML page and
publishes it with `publish_site`, which returns a public HTTPS URL. No account
or API key is needed for deed.page; the first publish returns an owner token
for that page.

From the repository root, run:

```sh
export OPENAI_API_KEY=your-openai-key
go run ./examples/mcp/deed_page -prompt 'A one-page cheat sheet for Go channels'
```

This uses `gpt-4o` to write the page and call `publish_site`, then prints the
URL. Model inference is billed separately by OpenAI. A tool filter exposes only
`publish_site` and `get_site` to the model, so it cannot delete or change other
sites. By default the page expires after 10 minutes (`-ttl 600`); pass `-ttl 0`
to keep it. Published pages are public, so don't publish anything private.

To try the round trip without a model or any API keys:

```sh
go run ./examples/mcp/deed_page -direct
```

Direct mode calls the MCP tools from Go: it publishes a small sample page,
fetches the URL, then deletes it with `delete_site` using the token from the
publish result. The token stays in memory and is never printed.

The example sets `User-Agent: openai-agents-go/deed-page-example` on MCP
requests, uses a 60-second HTTP timeout and a three-minute overall deadline, and
supports cancellation with Ctrl+C. Tracing is disabled for this example.

Run the offline tests, which use a local MCP fixture, with:

```sh
go test ./examples/mcp/deed_page
```

Docs: https://deed.page/docs
