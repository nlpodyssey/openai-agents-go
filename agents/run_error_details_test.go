package agents_test

import (
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunErrorIncludesData(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := agents.New("test").
		WithModelInstance(model).
		WithTools(agentstesting.GetFunctionTool("foo", "res"))

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("1"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	runner := agents.Runner{Config: agents.RunConfig{MaxTurns: 1}}
	_, err := runner.Run(t.Context(), agent, "hello")

	var target agents.MaxTurnsExceededError
	require.ErrorAs(t, err, &target)
	data := target.AgentsError.RunData
	require.NotNil(t, data)
	assert.Same(t, agent, data.LastAgent)
	assert.Len(t, data.RawResponses, 1)
	assert.NotEmpty(t, data.NewItems)
}

func TestStreamedRunErrorIncludesData(t *testing.T) {
	model := agentstesting.NewFakeModel(nil)
	agent := agents.New("test").
		WithModelInstance(model).
		WithTools(agentstesting.GetFunctionTool("foo", "res"))

	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("1"),
			agentstesting.GetFunctionToolCall("foo", `{"a": "b"}`),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		}},
	})

	runner := agents.Runner{Config: agents.RunConfig{MaxTurns: 1}}
	result, err := runner.RunStreamed(t.Context(), agent, "hello")
	require.NoError(t, err)

	err = result.StreamEvents(func(agents.StreamEvent) error { return nil })

	var target agents.MaxTurnsExceededError
	require.ErrorAs(t, err, &target)
	data := target.AgentsError.RunData
	require.NotNil(t, data)
	assert.Same(t, agent, data.LastAgent)
	assert.Len(t, data.RawResponses, 1)
	assert.NotEmpty(t, data.NewItems)
}
