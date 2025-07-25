package main

import (
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
)

const SearchAgentInstructions = "You are a research assistant. Given a search term, you search the web for that term and " +
	"produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 " +
	"words. Capture the main points. Write succinctly, no need to have complete sentences or good " +
	"grammar. This will be consumed by someone synthesizing a report, so its vital you capture the " +
	"essence and ignore any fluff. Do not include any additional commentary other than the summary " +
	"itself."

var SearchAgent = agents.New("Search agent").
	WithInstructions(SearchAgentInstructions).
	WithTools(agents.WebSearchTool{}).
	WithModel("gpt-4o").
	WithModelSettings(modelsettings.ModelSettings{
		ToolChoice: modelsettings.ToolChoiceRequired,
	})
