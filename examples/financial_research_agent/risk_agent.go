package main

import (
	"github.com/nlpodyssey/openai-agents-go/agents"
)

// A sub‑agent specializing in identifying risk factors or concerns.

const RiskPrompt = "You are a risk analyst looking for potential red flags in a company's outlook. " +
	"Given background research, produce a short analysis of risks such as competitive threats, " +
	"regulatory issues, supply chain problems, or slowing growth. Keep it under 2 paragraphs."

var RiskAgent = agents.New("RiskAnalystAgent").
	WithInstructions(RiskPrompt).
	WithOutputType(agents.OutputType[AnalysisSummary]()).
	WithModel("gpt-4o")
