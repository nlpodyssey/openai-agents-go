package agents_test

import (
	"context"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// PromptCaptureFakeModel extends agentstesting.FakeModel also recording the prompt passed to the model.
type PromptCaptureFakeModel struct {
	*agentstesting.FakeModel
	LastPrompt *responses.ResponsePromptParam
}

func NewPromptCaptureFakeModel() *PromptCaptureFakeModel {
	return &PromptCaptureFakeModel{
		FakeModel: agentstesting.NewFakeModel(false, nil),
	}
}

func (m *PromptCaptureFakeModel) GetResponse(ctx context.Context, params agents.ModelResponseParams) (*agents.ModelResponse, error) {
	// Record the prompt that the agent resolved and passed in.
	m.LastPrompt = &params.Prompt
	return m.FakeModel.GetResponse(ctx, params)
}

func TestStaticPromptIsResolvedCorrectly(t *testing.T) {
	staticPrompt := agents.Prompt{
		ID:      "my_prompt",
		Version: param.NewOpt("1"),
		Variables: map[string]responses.ResponsePromptVariableUnionParam{
			"some_var": {OfString: param.NewOpt("some_value")},
		},
	}

	agent := agents.New("test").WithPrompt(staticPrompt)

	resolved, ok, err := agent.GetPrompt(t.Context())
	require.NoError(t, err)
	assert.True(t, ok)
	assert.Equal(t, responses.ResponsePromptParam{
		ID:      "my_prompt",
		Version: param.NewOpt("1"),
		Variables: map[string]responses.ResponsePromptVariableUnionParam{
			"some_var": {OfString: param.NewOpt("some_value")},
		},
	}, resolved)
}

func TestDynamicPromptIsResolvedCorrectly(t *testing.T) {
	dynamicPromptValue := agents.Prompt{
		ID:      "dyn_prompt",
		Version: param.NewOpt("2"),
	}

	dynamicPromptFn := func(context.Context, *agents.Agent) (agents.Prompt, error) {
		return dynamicPromptValue, nil
	}

	agent := agents.New("test").WithPrompt(agents.DynamicPromptFunction(dynamicPromptFn))

	resolved, ok, err := agent.GetPrompt(t.Context())
	require.NoError(t, err)
	assert.True(t, ok)
	assert.Equal(t, responses.ResponsePromptParam{
		ID:        "dyn_prompt",
		Version:   param.NewOpt("2"),
		Variables: nil,
	}, resolved)
}

func TestPromptIsPassedToModel(t *testing.T) {
	staticPrompt := agents.Prompt{ID: "model_prompt"}

	model := NewPromptCaptureFakeModel()
	agent := agents.New("test").WithModelInstance(model).WithPrompt(staticPrompt)

	// Ensure the model returns a simple message so the run completes in one turn.
	model.SetNextOutput(agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{
			agentstesting.GetTextMessage("done"),
		},
	})

	_, err := agents.Run(t.Context(), agent, "hello")
	require.NoError(t, err)

	// The model should have received the prompt resolved by the agent.
	expectedPrompt := &responses.ResponsePromptParam{
		ID:        "model_prompt",
		Version:   param.Opt[string]{},
		Variables: nil,
	}

	assert.Equal(t, expectedPrompt, model.LastPrompt)
}
