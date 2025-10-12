package workflowrunner

import (
	"fmt"
	"strings"
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

type consolePrinter struct {
	enabled      bool
	verbose      bool
	startTime    time.Time
	turns        int
	tools        map[string]struct{}
	firstAgent   string
	lastAgent    string
	currentAgent string
	streamAgent  string
	streamActive bool
}

func newConsolePrinter(enabled bool, verbose bool) *consolePrinter {
	return &consolePrinter{
		enabled: enabled,
		verbose: verbose,
		tools:   make(map[string]struct{}),
	}
}

func (p *consolePrinter) OnRunStarted(query string) {
	if !p.enabled {
		return
	}
	p.startTime = time.Now()
	fmt.Printf("user: %s\n", shorten(strings.TrimSpace(query), 240))
}

func (p *consolePrinter) OnStreamEvent(event agents.StreamEvent) {
	if !p.enabled {
		return
	}
	switch ev := event.(type) {
	case agents.RawResponsesStreamEvent:
		p.handleRawEvent(ev)
	case agents.RunItemStreamEvent:
		p.handleRunItem(ev.Item)
	case agents.AgentUpdatedStreamEvent:
		if ev.NewAgent != nil {
			p.currentAgent = ev.NewAgent.Name
			if p.verbose {
				fmt.Printf("[agent switch] -> %s\n", ev.NewAgent.Name)
			}
		}
	}
}

func (p *consolePrinter) handleRawEvent(ev agents.RawResponsesStreamEvent) {
	switch ev.Data.Type {
	case "response.output_text.delta", "response.reasoning_summary_text.delta":
		if !p.streamActive {
			p.streamAgent = fallbackAgent(p.currentAgent)
			if p.firstAgent == "" && p.streamAgent != "" {
				p.firstAgent = p.streamAgent
			}
			if p.streamAgent != "" {
				p.lastAgent = p.streamAgent
			}
			p.turns++
			fmt.Printf("agent %s: ", p.streamAgent)
			p.streamActive = true
		}
		fmt.Print(ev.Data.Delta)
	case "response.output_text.done", "response.reasoning_summary_text.done", "response.content_part.done":
		if p.streamActive {
			fmt.Println()
			p.streamActive = false
			p.streamAgent = ""
		}
	}
}

func (p *consolePrinter) handleRunItem(item agents.RunItem) {
	switch v := item.(type) {
	case agents.MessageOutputItem:
		if p.streamActive {
			p.streamActive = false
			p.streamAgent = ""
			if v.Agent != nil {
				p.lastAgent = v.Agent.Name
			}
			return
		}
		agentName := displayAgentName(v.Agent)
		if agentName != "" {
			if p.firstAgent == "" {
				p.firstAgent = agentName
			}
			p.lastAgent = agentName
		}
		p.turns++
		message := strings.TrimSpace(agents.ItemHelpers().TextMessageOutput(v))
		if message != "" {
			fmt.Printf("agent %s: %s\n", agentName, shorten(message, 240))
		}
	case agents.ToolCallItem:
		toolName := readableToolName(v.RawItem)
		if toolName != "" {
			p.tools[toolName] = struct{}{}
		}
		fmt.Printf("tool %s called by %s\n", toolName, displayAgentName(v.Agent))
	case agents.ToolCallOutputItem:
		toolName := readableToolOutputName(v.RawItem)
		if toolName != "" {
			p.tools[toolName] = struct{}{}
		}
		output := shorten(stringifyValue(v.Output), 200)
		if output != "" {
			fmt.Printf("tool %s output: %s\n", toolName, output)
		}
	case agents.HandoffOutputItem:
		fmt.Printf("handoff %s -> %s\n", displayAgentName(v.SourceAgent), displayAgentName(v.TargetAgent))
	case agents.MCPApprovalRequestItem:
		fmt.Printf("approval required (%s): request %s for tool %s\n",
			displayAgentName(v.Agent), shorten(v.RawItem.ID, 40), v.RawItem.Name)
	case agents.MCPApprovalResponseItem:
		fmt.Printf("approval response: request %s approve=%t\n",
			shorten(v.RawItem.ApprovalRequestID, 40), v.RawItem.Approve)
	}
}

func (p *consolePrinter) OnRunCompleted(finalOutput any, lastAgent string) {
	if !p.enabled {
		return
	}
	if lastAgent != "" {
		p.lastAgent = lastAgent
	}
	duration := time.Since(p.startTime)
	fmt.Println("---")
	fmt.Println("Session summary")
	fmt.Printf("  turns: %d\n", p.turns)
	fmt.Printf("  starting agent: %s\n", p.firstAgent)
	fmt.Printf("  final agent: %s\n", p.lastAgent)
	fmt.Printf("  runtime: %s\n", duration.Truncate(time.Millisecond))
	if len(p.tools) > 0 {
		toolNames := make([]string, 0, len(p.tools))
		for name := range p.tools {
			toolNames = append(toolNames, name)
		}
		fmt.Printf("  tools: %s\n", strings.Join(toolNames, ", "))
	}
	if finalOutput != nil && p.verbose {
		fmt.Printf("  final output: %s\n", shorten(stringifyValue(finalOutput), 400))
	}
}

func (p *consolePrinter) OnRunFailed(err error) {
	if !p.enabled {
		return
	}
	duration := time.Since(p.startTime)
	fmt.Println("---")
	fmt.Printf("Run failed after %s: %v\n", duration.Truncate(time.Millisecond), err)
}

func readableToolName(raw agents.ToolCallItemType) string {
	switch v := raw.(type) {
	case agents.ResponseFunctionToolCall:
		return v.Name
	case agents.ResponseFunctionWebSearch:
		return "web_search"
	case agents.ResponseFileSearchToolCall:
		return "file_search"
	case agents.ResponseCodeInterpreterToolCall:
		return "code_interpreter"
	case agents.ResponseOutputItemImageGenerationCall:
		return "image_generation"
	case agents.ResponseOutputItemMcpCall:
		return v.ServerLabel
	default:
		return fmt.Sprintf("%T", raw)
	}
}

func readableToolOutputName(raw agents.ToolCallOutputRawItem) string {
	switch raw.(type) {
	case agents.ResponseInputItemFunctionCallOutputParam:
		return "function_call_output"
	case agents.ResponseInputItemComputerCallOutputParam:
		return "computer_call_output"
	case agents.ResponseInputItemLocalShellCallOutputParam:
		return "local_shell_output"
	default:
		return fmt.Sprintf("%T", raw)
	}
}

func stringifyValue(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

func shorten(s string, max int) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func fallbackAgent(name string) string {
	if strings.TrimSpace(name) == "" {
		return "agent"
	}
	return name
}
