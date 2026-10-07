package agents

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/require"
)

func TestTResponseInputItemFromToolCallItemType_FileSearchCall(t *testing.T) {
	input := ResponseFileSearchToolCall(responses.ResponseFileSearchToolCall{
		ID:      "fs_1",
		Queries: []string{"hello"},
	})

	require.NotPanics(t, func() {
		out := TResponseInputItemFromToolCallItemType(input)
		require.NotNil(t, out.OfFileSearchCall)
		require.Equal(t, "fs_1", out.OfFileSearchCall.ID)
	})
}

func TestTResponseInputItemFromToolCallItemType_WebSearchCallPreservesQueriesAndSources(t *testing.T) {
	input := ResponseFunctionWebSearch(responses.ResponseFunctionWebSearch{
		ID: "ws_1",
		Action: responses.ResponseFunctionWebSearchActionUnion{
			Type: "search", Queries: []string{"first", "second"},
			Sources: []responses.ResponseFunctionWebSearchActionSearchSource{{
				Type: "url", URL: "https://example.com/source",
			}},
		},
	})

	out := TResponseInputItemFromToolCallItemType(input)
	search := out.OfWebSearchCall.Action.OfSearch
	require.NotNil(t, search)
	require.False(t, search.Query.Valid())
	require.Equal(t, []string{"first", "second"}, search.Queries)
	require.Equal(t, "https://example.com/source", search.Sources[0].URL)
	encoded, err := json.Marshal(search)
	require.NoError(t, err)
	require.False(t, strings.Contains(string(encoded), `"query":`), string(encoded))
}

func TestTResponseInputItemFromToolCallItemType_WebSearchCallPreservesFindInPage(t *testing.T) {
	input := ResponseFunctionWebSearch(responses.ResponseFunctionWebSearch{
		ID: "ws_1",
		Action: responses.ResponseFunctionWebSearchActionUnion{
			Type: "find_in_page", Pattern: "needle", URL: "https://example.com/page",
		},
	})

	out := TResponseInputItemFromToolCallItemType(input)
	find := out.OfWebSearchCall.Action.OfFind
	require.NotNil(t, find)
	require.Equal(t, "needle", find.Pattern)
	require.Equal(t, "https://example.com/page", find.URL)
}

func TestTResponseInputItemFromToolCallItemType_WebSearchCall(t *testing.T) {
	input := ResponseFunctionWebSearch(responses.ResponseFunctionWebSearch{
		ID: "ws_1",
		Action: responses.ResponseFunctionWebSearchActionUnion{
			Type:  "search",
			Query: "hello",
		},
	})

	require.NotPanics(t, func() {
		out := TResponseInputItemFromToolCallItemType(input)
		require.NotNil(t, out.OfWebSearchCall)
		require.Equal(t, "ws_1", out.OfWebSearchCall.ID)
		require.NotNil(t, out.OfWebSearchCall.Action.OfSearch)
		require.Equal(t, "hello", out.OfWebSearchCall.Action.OfSearch.Query.Or(""))
	})
}
