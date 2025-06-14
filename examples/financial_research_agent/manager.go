package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

// summaryExtractor is a custom output extractor for sub‑agents that return an AnalysisSummary.
func summaryExtractor(runResult agents.RunResult) (string, error) {
	// The financial/risk analyst agents emit an AnalysisSummary with a `Summary` field.
	// We want the tool call to return just that summary text so the writer can drop it inline.
	return runResult.FinalOutput.(AnalysisSummary).Summary, nil
}

// FinancialResearchManager orchestrates the full flow: planning, searching,
// sub‑analysis, writing, and verification.
type FinancialResearchManager struct{}

func NewFinancialResearchManager() *FinancialResearchManager {
	return new(FinancialResearchManager)
}

func (frm FinancialResearchManager) Run(ctx context.Context, query string) error {
	fmt.Println("Starting financial research...")

	searchPlan, err := frm.planSearches(ctx, query)
	if err != nil {
		return err
	}

	searchResults, err := frm.performSearches(ctx, *searchPlan)
	if err != nil {
		return err
	}

	report, err := frm.writeReport(ctx, query, searchResults)
	if err != nil {
		return err
	}

	verification, err := frm.verifyReport(ctx, *report)
	if err != nil {
		return err
	}

	fmt.Printf("Report summary\n\n%s\n", report.ShortSummary)
	fmt.Printf("\n\n=====REPORT=====\n\n\n")
	fmt.Printf("Report:\n%s\n", report.MarkdownReport)
	fmt.Printf("\n\n=====FOLLOW UP QUESTIONS=====\n\n\n")
	fmt.Println(strings.Join(report.FollowUpQuestions, "\n"))
	fmt.Printf("\n\n=====VERIFICATION=====\n\n\n")
	fmt.Printf("%+v\n", *verification)

	return nil
}

func (frm FinancialResearchManager) planSearches(ctx context.Context, query string) (*FinancialSearchPlan, error) {
	fmt.Println("Planning searches...")
	result, err := agents.Run(ctx, PlannerAgent, "Query: "+query)
	if err != nil {
		return nil, err
	}
	searchPlan := result.FinalOutput.(FinancialSearchPlan)
	fmt.Printf("Will perform %d searches\n", len(searchPlan.Searches))
	return &searchPlan, nil
}

func (frm FinancialResearchManager) performSearches(ctx context.Context, searchPlan FinancialSearchPlan) ([]string, error) {
	fmt.Println("Searching...")

	var mu sync.Mutex
	var wg sync.WaitGroup
	numCompleted := 0
	results := make([]string, len(searchPlan.Searches))
	searchErrors := make([]error, len(searchPlan.Searches))

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	wg.Add(len(searchPlan.Searches))
	for i, item := range searchPlan.Searches {
		go func() {
			defer wg.Done()
			results[i], searchErrors[i] = frm.search(childCtx, item)
			if searchErrors[i] != nil {
				cancel()
			}

			mu.Lock()
			defer mu.Unlock()
			numCompleted += 1
			fmt.Printf("Searching... %d/%d completed\n", numCompleted, len(searchPlan.Searches))
		}()
	}

	wg.Wait()
	if err := errors.Join(searchErrors...); err != nil {
		return nil, err
	}
	return results, nil
}

func (frm FinancialResearchManager) search(ctx context.Context, item FinancialSearchItem) (string, error) {
	inputData := fmt.Sprintf("Search term: %s\nReason: %s", item.Query, item.Reason)
	result, err := agents.Run(ctx, SearchAgent, inputData)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s", result.FinalOutput), nil
}

func (frm FinancialResearchManager) writeReport(ctx context.Context, query string, searchResults []string) (*FinancialReportData, error) {
	// Expose the specialist analysts as tools so the writer can invoke them inline
	// and still produce the final FinancialReportData output.

	fundamentalsTool := FinancialsAgent.AsTool(agents.AgentAsToolParams{
		ToolName:              "fundamentals_analysis",
		ToolDescription:       "Use to get a short write‑up of key financial metrics",
		CustomOutputExtractor: summaryExtractor,
	})

	riskTool := RiskAgent.AsTool(agents.AgentAsToolParams{
		ToolName:              "risk_analysis",
		ToolDescription:       "Use to get a short write‑up of potential red flags",
		CustomOutputExtractor: summaryExtractor,
	})

	// Clone the agent and attach tools.
	writerWithTools := new(agents.Agent)
	*writerWithTools = *WriterAgent
	writerWithTools.Tools = []agents.Tool{fundamentalsTool, riskTool}

	fmt.Println("Generating report...")

	input := fmt.Sprintf("Original query: %s\nSummarized search results: %v", query, searchResults)
	result, err := agents.Run(ctx, writerWithTools, input)
	if err != nil {
		return nil, err
	}
	reportData := result.FinalOutput.(FinancialReportData)
	return &reportData, nil
}

func (frm FinancialResearchManager) verifyReport(ctx context.Context, report FinancialReportData) (*VerificationResult, error) {
	fmt.Println("Verifying report...")
	result, err := agents.Run(ctx, VerifierAgent, report.MarkdownReport)
	if err != nil {
		return nil, err
	}
	verificationResult := result.FinalOutput.(VerificationResult)
	return &verificationResult, nil
}
