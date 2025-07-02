package agents_test

import (
	"strings"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/openaitypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunDemoLoopRW(t *testing.T) {
	model := agentstesting.NewFakeModel(false, nil)
	model.AddMultipleTurnOutputs([]agentstesting.FakeModelTurnOutput{
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("hello"),
		}},
		{Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("good"),
		}},
	})
	agent := agents.New("test").WithModelInstance(model)

	inputs := strings.Join([]string{"Hi", "How are you?", "quit"}, "\n")
	r := strings.NewReader(inputs)
	var sb strings.Builder

	err := agents.RunDemoLoopRW(t.Context(), agent, false, r, &sb)
	require.NoError(t, err)

	output := sb.String()
	assert.Contains(t, output, "hello")
	assert.Contains(t, output, "good")
	assert.Equal(t, agents.InputItems{
		agentstesting.GetTextInputItem("Hi"),
		openaitypes.ResponseInputItemUnionParamFromResponseOutputItemUnion(agentstesting.GetTextMessage("hello")),
		agentstesting.GetTextInputItem("How are you?"),
	}, model.LastTurnArgs.Input)
}
