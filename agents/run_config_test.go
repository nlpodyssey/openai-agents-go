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

package agents_test

import (
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// DummyProvider is a simple model provider that always returns the same model,
// and records the model name it was asked to provide.
type DummyProvider struct {
	ModelToReturn agents.Model
	LastRequested *string
}

func NewDummyProvider(modelToReturn agents.Model) *DummyProvider {
	if modelToReturn == nil {
		modelToReturn = agentstesting.NewFakeModel(false, nil)
	}
	return &DummyProvider{ModelToReturn: modelToReturn}
}

func (dp *DummyProvider) GetModel(modelName string) (agents.Model, error) {
	// record the requested model name and return our test model
	if modelName != "" {
		dp.LastRequested = &modelName
	} else {
		dp.LastRequested = nil
	}
	return dp.ModelToReturn, nil
}

func TestModelProviderOnRunConfigIsUsedForAgentModelName(t *testing.T) {
	// When the agent's `model` attribute is a string and no explicit model override is
	// provided in the `RunConfig`, the `Runner` should resolve the model using the
	// `ModelProvider` on the `RunConfig`.
	fakeModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("from-provider"),
		},
	})
	provider := NewDummyProvider(fakeModel)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModelName("test-model")),
	}
	runConfig := agents.RunConfig{
		ModelProvider: provider,
	}
	result, err := (agents.Runner{Config: runConfig}).Run(
		t.Context(), agent, "any")
	require.NoError(t, err)
	// We picked up the model from our dummy provider
	require.NotNil(t, provider.LastRequested)
	assert.Equal(t, "test-model", *provider.LastRequested)
	assert.Equal(t, "from-provider", result.FinalOutput)
}

func TestRunConfigModelNameOverrideTakesPrecedence(t *testing.T) {
	// When a model name string is set on the RunConfig, then that name should be looked up
	// using the RunConfig's ModelProvider, and should override any model on the agent.
	fakeModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("from-provider"),
		},
	})
	provider := NewDummyProvider(fakeModel)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModelName("agent-model")),
	}
	runConfig := agents.RunConfig{
		Model:         param.NewOpt(agents.NewAgentModelName("override-name")),
		ModelProvider: provider,
	}
	result, err := (agents.Runner{Config: runConfig}).Run(
		t.Context(), agent, "any")
	require.NoError(t, err)
	// We should have requested the override name, not the agent.model
	require.NotNil(t, provider.LastRequested)
	assert.Equal(t, "override-name", *provider.LastRequested)
	assert.Equal(t, "from-provider", result.FinalOutput)
}

func TestRunConfigModelOverrideObjectTakesPrecedence(t *testing.T) {
	// When a concrete Model instance is set on the RunConfig, then that instance should be
	// returned by Runner.getModel regardless of the agent's model.
	fakeModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("override-object"),
		},
	})
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModelName("agent-model")),
	}
	runConfig := agents.RunConfig{
		Model: param.NewOpt(agents.NewAgentModel(fakeModel)),
	}
	result, err := (agents.Runner{Config: runConfig}).Run(
		t.Context(), agent, "any")
	require.NoError(t, err)
	// Our FakeModel on the RunConfig should have been used.
	assert.Equal(t, "override-object", result.FinalOutput)
}

func TestAgentModelObjectIsUsedWhenPresent(t *testing.T) {
	// If the agent has a concrete Model object set as its model, and the RunConfig does
	// not specify a model override, then that object should be used directly without
	// consulting the RunConfig's ModelProvider.
	fakeModel := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("from-agent-object"),
		},
	})
	provider := NewDummyProvider(nil)
	agent := &agents.Agent{
		Name:  "test",
		Model: param.NewOpt(agents.NewAgentModel(fakeModel)),
	}
	runConfig := agents.RunConfig{
		ModelProvider: provider,
	}
	result, err := (agents.Runner{Config: runConfig}).Run(
		t.Context(), agent, "any")
	require.NoError(t, err)
	// The dummy provider should never have been called, and the output should come from
	// the FakeModel on the agent.
	assert.Nil(t, provider.LastRequested)
	assert.Equal(t, "from-agent-object", result.FinalOutput)
}
