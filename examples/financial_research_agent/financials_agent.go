package main

import (
	"github.com/nlpodyssey/openai-agents-go/agents"
)

// A sub‑agent focused on analyzing a company's fundamentals.

const FinancialsPrompt = "You are a financial analyst focused on company fundamentals such as revenue, " +
	"profit, margins and growth trajectory. Given a collection of web (and optional file) " +
	"search results about a company, write a concise analysis of its recent financial " +
	"performance. Pull out key metrics or quotes. Keep it under 2 paragraphs."

var FinancialsAgent = agents.New("FundamentalsAnalystAgent").
	WithInstructions(FinancialsPrompt).
	WithOutputSchema(AnalysisSummarySchema{}).
	WithModel("gpt-4o")
