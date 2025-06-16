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

package agentstesting

import (
	"context"
	"iter"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
)

type FakeModel struct {
	TurnOutputs    []FakeModelTurnOutput
	LastTurnArgs   FakeModelLastTurnArgs
	HardcodedUsage *usage.Usage
}

type FakeModelTurnOutput struct {
	Value []agents.TResponseOutputItem
	Error error
}

type FakeModelLastTurnArgs struct {
	SystemInstructions param.Opt[string]
	Input              agents.Input
	ModelSettings      modelsettings.ModelSettings
	Tools              []agents.Tool
	OutputSchema       agents.AgentOutputSchemaInterface
	// optional
	PreviousResponseID string
}

func NewFakeModel(initialOutput *FakeModelTurnOutput) *FakeModel {
	m := &FakeModel{}
	if initialOutput != nil && (len(initialOutput.Value) > 0 || initialOutput.Error != nil) {
		m.TurnOutputs = []FakeModelTurnOutput{*initialOutput}
	}
	return m
}

func (m *FakeModel) SetHardcodedUsage(u usage.Usage) {
	m.HardcodedUsage = &u
}

func (m *FakeModel) SetNextOutput(output FakeModelTurnOutput) {
	m.TurnOutputs = append(m.TurnOutputs, output)
}

func (m *FakeModel) AddMultipleTurnOutputs(outputs []FakeModelTurnOutput) {
	m.TurnOutputs = append(m.TurnOutputs, outputs...)
}

func (m *FakeModel) GetNextOutput() FakeModelTurnOutput {
	if len(m.TurnOutputs) == 0 {
		return FakeModelTurnOutput{}
	}
	v := m.TurnOutputs[0]
	m.TurnOutputs = m.TurnOutputs[1:]
	return v
}

func (m *FakeModel) GetResponse(_ context.Context, params agents.ModelResponseParams) (*agents.ModelResponse, error) {
	m.LastTurnArgs = FakeModelLastTurnArgs{
		SystemInstructions: params.SystemInstructions,
		Input:              params.Input,
		ModelSettings:      params.ModelSettings,
		Tools:              params.Tools,
		OutputSchema:       params.OutputSchema,
		PreviousResponseID: params.PreviousResponseID,
	}

	output := m.GetNextOutput()

	if output.Error != nil {
		return nil, output.Error
	}

	u := m.HardcodedUsage
	if u == nil {
		u = usage.NewUsage()
	}

	return &agents.ModelResponse{
		Output:     output.Value,
		Usage:      u,
		ResponseID: "",
	}, nil
}

func (m *FakeModel) StreamResponse(_ context.Context, params agents.ModelResponseParams) (iter.Seq2[*agents.TResponseStreamEvent, error], error) {
	m.LastTurnArgs = FakeModelLastTurnArgs{
		SystemInstructions: params.SystemInstructions,
		Input:              params.Input,
		ModelSettings:      params.ModelSettings,
		Tools:              params.Tools,
		OutputSchema:       params.OutputSchema,
		PreviousResponseID: params.PreviousResponseID,
	}

	output := m.GetNextOutput()

	return func(yield func(*agents.TResponseStreamEvent, error) bool) {
		if output.Error != nil {
			yield(nil, output.Error)
			return
		}

		u := m.HardcodedUsage
		if u == nil {
			u = usage.NewUsage()
		}

		yield(&agents.TResponseStreamEvent{ // responses.ResponseCompletedEvent
			Response: GetResponseObj(output.Value, "", m.HardcodedUsage),
			Type:     "response.completed",
		}, nil)
	}, nil
}

func GetResponseObj(
	output []agents.TResponseOutputItem,
	responseID string,
	u *usage.Usage,
) responses.Response {
	if responseID == "" {
		responseID = "123"
	}

	var responseUsage responses.ResponseUsage
	if u != nil {
		responseUsage = responses.ResponseUsage{
			InputTokens: int64(u.InputTokens),
			InputTokensDetails: responses.ResponseUsageInputTokensDetails{
				CachedTokens: 0,
			},
			OutputTokens: int64(u.OutputTokens),
			OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
				ReasoningTokens: 0,
			},
			TotalTokens: int64(u.TotalTokens),
		}
	}

	return responses.Response{
		ID:        responseID,
		CreatedAt: 123,
		Model:     "test_model",
		Object:    "response",
		Output:    output,
		ToolChoice: responses.ResponseToolChoiceUnion{
			OfToolChoiceMode: responses.ToolChoiceOptionsNone,
		},
		Tools:             nil,
		TopP:              0,
		ParallelToolCalls: false,
		Usage:             responseUsage,
	}
}
