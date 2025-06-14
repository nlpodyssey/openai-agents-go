package main

import (
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/tools"
)

// Given a search term, use web search to pull back a brief summary.
// Summaries should be concise but capture the main financial points.

const SearchInstructions = "You are a research assistant specializing in financial topics. " +
	"Given a search term, use web search to retrieve up‑to‑date context and " +
	"produce a short summary of at most 300 words. Focus on key numbers, events, " +
	"or quotes that will be useful to a financial analyst."

var SearchAgent = agents.New("FinancialSearchAgent").
	WithInstructions(SearchInstructions).
	WithTools(tools.WebSearchTool{}).
	WithModelSettings(modelsettings.ModelSettings{
		ToolChoice: "required",
	}).
	WithModel("gpt-4o")
