package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

type ResearchManager struct{}

func NewResearchManager() *ResearchManager {
	return new(ResearchManager)
}

func (rm *ResearchManager) Run(ctx context.Context, query string) error {
	fmt.Println("Starting research...")

	searchPlan, err := rm.planSearches(ctx, query)
	if err != nil {
		return err
	}

	searchResults, err := rm.performSearches(ctx, *searchPlan)
	if err != nil {
		return err
	}

	report, err := rm.writeReport(ctx, query, searchResults)
	if err != nil {
		return err
	}

	fmt.Printf("Report summary\n\n%s\n", report.ShortSummary)
	fmt.Print("\n\n=====REPORT=====\n\n\n")
	fmt.Printf("Report: %s\n", report.MarkdownReport)
	fmt.Print("\n\n=====FOLLOW UP QUESTIONS=====\n\n\n")
	followUpQuestions := strings.Join(report.FollowUpQuestions, "\n")
	fmt.Printf("Follow up questions: %s", followUpQuestions)

	return nil
}

func (rm *ResearchManager) planSearches(ctx context.Context, query string) (*WebSearchPlan, error) {
	fmt.Println("Planning searches...")
	result, err := agents.Run(ctx, PlannerAgent, "Query: "+query)
	if err != nil {
		return nil, err
	}
	searchPlan := result.FinalOutput.(WebSearchPlan)
	fmt.Printf("Will perform %d searches\n", len(searchPlan.Searches))
	return &searchPlan, nil
}

func (rm *ResearchManager) performSearches(ctx context.Context, searchPlan WebSearchPlan) ([]string, error) {
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
			results[i], searchErrors[i] = rm.search(childCtx, item)
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

func (rm *ResearchManager) search(ctx context.Context, item WebSearchItem) (string, error) {
	input := fmt.Sprintf("Search term: %s\nReason for searching: %s", item.Query, item.Reason)
	result, err := agents.Run(ctx, SearchAgent, input)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s", result.FinalOutput), nil
}

func (rm *ResearchManager) writeReport(ctx context.Context, query string, searchResults []string) (*ReportData, error) {
	fmt.Println("Generating report...")

	input := fmt.Sprintf("Original query: %s\nSummarized search results: %v", query, searchResults)
	result, err := agents.Run(ctx, WriterAgent, input)
	if err != nil {
		return nil, err
	}
	reportData := result.FinalOutput.(ReportData)
	return &reportData, nil
}
