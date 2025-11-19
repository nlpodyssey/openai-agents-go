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
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strings"

	"github.com/google/uuid"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agents/extensions/handoff_prompt"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

////// CONTEXT

type airlineAgentContextKey struct{}

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

func FAQLookup(_ context.Context, args FAQLookupArgs) (string, error) {
	question := strings.ToLower(args.Question)
	contains := func(values ...string) bool {
		for _, v := range values {
			if strings.Contains(question, v) {
				return true
			}
		}
		return false
	}

	switch {
	case contains("bag", "baggage", "luggage", "carry-on", "hand luggage", "hand carry"):
		return "You are allowed to bring one bag on the plane. " +
			"It must be under 50 pounds and 22 inches x 14 inches x 9 inches.", nil
	case contains("seat", "seats", "seating", "plane"):
		return "There are 120 seats on the plane. " +
			"There are 22 business class seats and 98 economy seats. " +
			"Exit rows are rows 4 and 16. " +
			"Rows 5-8 are Economy Plus, with extra legroom. ", nil
	case contains("wifi", "internet", "wireless", "connectivity", "network", "online"):
		return "We have free wifi on the plane, join Airline-Wifi", nil
	default:
		return "I'm sorry, I don't know the answer to that question.", nil
	}
}

var FAQLookupTool = agents.NewFunctionTool("faq_lookup_tool", "Lookup frequently asked questions.", FAQLookup)

type UpdateSeatArgs struct {
	ConfirmationNumber string `json:"confirmation_number" jsonschema_description:"The confirmation number for the flight."`
	NewSeat            string `json:"new_seat" jsonschema_description:"The new seat to update to."`
}

func UpdateSeat(ctx context.Context, args UpdateSeatArgs) (string, error) {
	airlineCtx := ctx.Value(airlineAgentContextKey{}).(*AirlineAgentContext)

	// Update the context based on the customer's input
	airlineCtx.ConfirmationNumber = args.ConfirmationNumber
	airlineCtx.SeatNumber = args.NewSeat

	// Ensure that the flight number has been set by the incoming handoff
	if airlineCtx.FlightNumber == "" {
		return "", errors.New("flight number is required")
	}
	return fmt.Sprintf(
		"Updated seat to %s for confirmation number %s",
		args.NewSeat, args.ConfirmationNumber,
	), nil
}

var UpdateSeatTool = agents.NewFunctionTool("update_seat", "Update the seat for a given confirmation number.", UpdateSeat)

////// HOOKS

func OnSeatBookingHandoff(ctx context.Context) error {
	flightNumber := fmt.Sprintf("FLT-%d", rand.Intn(900)+100)
	airlineCtx := ctx.Value(airlineAgentContextKey{}).(*AirlineAgentContext)
	airlineCtx.FlightNumber = flightNumber
	return nil
}

////// AGENTS

const Model = "gpt-4o"

var (
	FAQAgent = agents.New("FAQ Agent").
			WithHandoffDescription("A helpful agent that can answer questions about the airline.").
			WithInstructions(handoff_prompt.PromptWithHandoffInstructions(
			`You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
Use the following routine to support the customer.
# Routine
1. Identify the last question asked by the customer.
2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
3. If you cannot answer the question, transfer back to the triage agent.`)).
		WithTools(FAQLookupTool).
		WithModel(Model)

	SeatBookingAgent = agents.New("Seat Booking Agent").
				WithHandoffDescription("A helpful agent that can update a seat on a flight.").
				WithInstructions(handoff_prompt.PromptWithHandoffInstructions(
			`You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
Use the following routine to support the customer.
# Routine
1. Ask for their confirmation number.
2. Ask the customer what their desired seat number is.
3. Use the update seat tool to update the seat on the flight.
If the customer asks a question that is not related to the routine, transfer back to the triage agent.`)).
		WithTools(UpdateSeatTool).
		WithModel(Model)

	TriageAgent = agents.New("Triage Agent").
			WithHandoffDescription("A triage agent that can delegate a customer's request to the appropriate agent.").
			WithInstructions(handoff_prompt.PromptWithHandoffInstructions(
			`You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents.`)).
		WithAgentHandoffs(FAQAgent).
		WithHandoffs(
			agents.HandoffFromAgent(agents.HandoffFromAgentParams{
				Agent:     SeatBookingAgent,
				OnHandoff: agents.OnHandoffWithoutInput(OnSeatBookingHandoff),
			}),
		).
		WithModel(Model)
)

func init() {
	FAQAgent.AgentHandoffs = append(FAQAgent.AgentHandoffs, TriageAgent)
	SeatBookingAgent.AgentHandoffs = append(SeatBookingAgent.AgentHandoffs, TriageAgent)
}

////// RUN

func main() {
	currentAgent := TriageAgent
	var inputItems []agents.TResponseInputItem
	ctx := context.WithValue(context.Background(), airlineAgentContextKey{}, new(AirlineAgentContext))

	// Normally, each input from the user would be an API request to your app,
	// and you can wrap the request in a RunTrace().
	// Here, we'll just use a random UUID for the conversation ID
	u := uuid.New()
	conversationID := hex.EncodeToString(u[:])[:16]

	for {
		fmt.Print("Enter your message: ")
		_ = os.Stdout.Sync()
		line, _, err := bufio.NewReader(os.Stdin).ReadLine()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			panic(err)
		}
		userInput := string(line)

		err = tracing.RunTrace(
			ctx, tracing.TraceParams{WorkflowName: "Customer service", GroupID: conversationID},
			func(ctx context.Context, _ tracing.Trace) error {
				inputItems = append(inputItems, agents.TResponseInputItem{
					OfMessage: &responses.EasyInputMessageParam{
						Content: responses.EasyInputMessageContentUnionParam{
							OfString: param.NewOpt(userInput),
						},
						Role: responses.EasyInputMessageRoleUser,
						Type: responses.EasyInputMessageTypeMessage,
					},
				})

				result, err := agents.RunInputs(ctx, currentAgent, inputItems)
				if err != nil {
					return err
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
				currentAgent = result.LastAgent
				return nil
			},
		)
		if err != nil {
			panic(err)
		}
	}
}
