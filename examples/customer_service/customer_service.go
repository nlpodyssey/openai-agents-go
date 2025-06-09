// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"strings"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agents/extensions/handoff_prompt"
	"github.com/nlpodyssey/openai-agents-go/runcontext"
	"github.com/nlpodyssey/openai-agents-go/tools"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

////// CONTEXT

type AirlineAgentContext struct {
	PassengerName      string
	ConfirmationNumber string
	SeatNumber         string
	FlightNumber       string
}

////// TOOLS

type FAQLookupArgs struct {
	Question string `json:"question"`
}

func FAQLookup(args FAQLookupArgs) string {
	q := args.Question
	switch {
	case strings.Contains(q, "bag") || strings.Contains(q, "baggage"):
		return "You are allowed to bring one bag on the plane. " +
			"It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
	case strings.Contains(q, "seats") || strings.Contains(q, "plane"):
		return "There are 120 seats on the plane. " +
			"There are 22 business class seats and 98 economy seats. " +
			"Exit rows are rows 4 and 16. " +
			"Rows 5-8 are Economy Plus, with extra legroom. "
	case strings.Contains(q, "wifi"):
		return "We have free wifi on the plane, join Airline-Wifi"
	default:
		return "I'm sorry, I don't know the answer to that question."
	}
}

var FAQLookupTool = tools.Function{
	Name:        "faq_lookup_tool",
	Description: "Lookup frequently asked questions.",
	ParamsJSONSchema: map[string]any{
		"title":                "faq_lookup_tool_args",
		"type":                 "object",
		"required":             []string{"question"},
		"additionalProperties": false,
		"properties": map[string]any{
			"question": map[string]any{
				"title": "Question",
				"type":  "string",
			},
		},
	},
	OnInvokeTool: func(_ context.Context, _ *runcontext.Wrapper, arguments string) (any, error) {
		var args FAQLookupArgs
		err := json.Unmarshal([]byte(arguments), &args)
		if err != nil {
			return nil, err
		}
		return FAQLookup(args), nil
	},
}

type UpdateSeatArgs struct {
	ConfirmationNumber string `json:"confirmation_number"`
	NewSeat            string `json:"new_seat"`
}

func UpdateSeat(cw *runcontext.Wrapper, args UpdateSeatArgs) (string, error) {
	ctx := cw.Context.(*AirlineAgentContext)

	// Update the context based on the customer's input
	ctx.ConfirmationNumber = args.ConfirmationNumber
	ctx.SeatNumber = args.NewSeat

	// Ensure that the flight number has been set by the incoming handoff
	if ctx.FlightNumber == "" {
		return "", errors.New("flight number is required")
	}
	return fmt.Sprintf(
		"Updated seat to %s for confirmation number %s",
		args.NewSeat, args.ConfirmationNumber,
	), nil
}

var UpdateSeatTool = tools.Function{
	Name:        "update_seat",
	Description: "Update the seat for a given confirmation number.",
	ParamsJSONSchema: map[string]any{
		"title":                "update_seat_args",
		"type":                 "object",
		"required":             []string{"confirmation_number", "new_seat"},
		"additionalProperties": false,
		"properties": map[string]any{
			"confirmation_number": map[string]any{
				"title":       "Confirmation number",
				"description": "The confirmation number for the flight.",
				"type":        "string",
			},
			"new_seat": map[string]any{
				"title":       "New seat",
				"description": "The new seat to update to.",
				"type":        "string",
			},
		},
	},
	OnInvokeTool: func(_ context.Context, cw *runcontext.Wrapper, arguments string) (any, error) {
		var args UpdateSeatArgs
		err := json.Unmarshal([]byte(arguments), &args)
		if err != nil {
			return nil, err
		}
		return UpdateSeat(cw, args)
	},
}

////// HOOKS

func OnSeatBookingHandoff(_ context.Context, cw *runcontext.Wrapper) error {
	flightNumber := fmt.Sprintf("FLT-%d", rand.Intn(900)+100)
	ctx := cw.Context.(*AirlineAgentContext)
	ctx.FlightNumber = flightNumber
	return nil
}

////// AGENTS

var (
	Model = agents.NewAgentModelName("gpt-4o")

	FAQAgent = &agents.Agent{
		Name:               "FAQ Agent",
		HandoffDescription: "A helpful agent that can answer questions about the airline.",
		Instructions: agents.InstructionsStr(
			handoff_prompt.RecommendedPromptPrefix + `
You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
Use the following routine to support the customer.
# Routine
1. Identify the last question asked by the customer.
2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
3. If you cannot answer the question, transfer back to the triage agent.`),
		Tools: []tools.Tool{FAQLookupTool},
		Model: param.NewOpt(Model),
	}

	SeatBookingAgent = &agents.Agent{
		Name:               "Seat Booking Agent",
		HandoffDescription: "A helpful agent that can update a seat on a flight.",
		Instructions: agents.InstructionsStr(
			handoff_prompt.RecommendedPromptPrefix + `
You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
Use the following routine to support the customer.
# Routine
1. Ask for their confirmation number.
2. Ask the customer what their desired seat number is.
3. Use the update seat tool to update the seat on the flight.
If the customer asks a question that is not related to the routine, transfer back to the triage agent.`),
		Tools: []tools.Tool{UpdateSeatTool},
		Model: param.NewOpt(Model),
	}

	TriageAgent = &agents.Agent{
		Name:               "Triage Agent",
		HandoffDescription: "A triage agent that can delegate a customer's request to the appropriate agent.",
		Instructions: agents.InstructionsStr(
			handoff_prompt.RecommendedPromptPrefix + `
You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents.`),
		AgentHandoffs: []*agents.Agent{FAQAgent},
		Handoffs: []agents.Handoff{
			agents.UnsafeHandoffFromAgent(agents.HandoffFromAgentParams{
				Agent:     SeatBookingAgent,
				OnHandoff: agents.OnHandoffWithoutInput(OnSeatBookingHandoff),
			}),
		},
		Model: param.NewOpt(Model),
	}
)

func init() {
	FAQAgent.AgentHandoffs = append(FAQAgent.AgentHandoffs, TriageAgent)
	SeatBookingAgent.AgentHandoffs = append(SeatBookingAgent.AgentHandoffs, TriageAgent)
}

////// RUN

func main() {
	currentAgent := TriageAgent
	var inputItems []agents.TResponseInputItem
	ctx := new(AirlineAgentContext)

	for {
		fmt.Print("Enter your message: ")
		_ = os.Stdout.Sync()
		line, _, err := bufio.NewReader(os.Stdin).ReadLine()
		if err != nil {
			panic(err)
		}
		userInput := string(line)

		inputItems = append(inputItems, agents.TResponseInputItem{
			OfMessage: &responses.EasyInputMessageParam{
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: param.NewOpt(userInput),
				},
				Role: responses.EasyInputMessageRoleUser,
				Type: responses.EasyInputMessageTypeMessage,
			},
		})

		result, err := agents.Runner().Run(context.Background(), agents.RunParams{
			StartingAgent: currentAgent,
			Input:         agents.InputItems(inputItems),
			Context:       ctx,
		})
		if err != nil {
			panic(err)
		}

		for _, newItem := range result.NewItems {
			switch newItem := newItem.(type) {
			case agents.MessageOutputItem:
				fmt.Printf("%s: %s\n", newItem.Agent.Name, agents.ItemHelpers().TextMessageOutput(newItem))
			case agents.HandoffOutputItem:
				fmt.Printf("Handed off from %s to %s\n", newItem.SourceAgent.Name, newItem.TargetAgent.Name)
			case agents.ToolCallItem:
				fmt.Printf("%s: Calling a tool\n", newItem.Agent.Name)
			case agents.ToolCallOutputItem:
				fmt.Printf("%s: Tool call output: %v\n", newItem.Agent.Name, newItem.Output)
			case agents.HandoffCallItem:
				fmt.Printf("%s: Skipping item: HandoffCallItem\n", newItem.Agent.Name)
			case agents.ReasoningItem:
				fmt.Printf("%s: Skipping item: ReasoningItem\n", newItem.Agent.Name)
			default:
				panic(fmt.Errorf("unexpected item type %T\n", newItem))
			}
		}

		inputItems = result.ToInputList()
		currentAgent = result.LastAgent()
	}
}
