package agents_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCallModelInputFilter(t *testing.T) {
	t.Run("non streamed", func(t *testing.T) {
		model := agentstesting.NewFakeModel(false, nil)
		agent := agents.New("test").WithModelInstance(model)

		// Prepare model output
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("ok"),
			},
		})

		filterFn := func(_ context.Context, data agents.CallModelData) (*agents.ModelInputData, error) {
			input := append(slices.Clone(data.ModelData.Input), agentstesting.GetTextInputItem("added-sync"))
			return &agents.ModelInputData{
				Input:        input,
				Instructions: param.NewOpt("filtered-sync"),
			}, nil
		}

		runner := agents.Runner{
			Config: agents.RunConfig{
				CallModelInputFilter: filterFn,
			},
		}
		_, err := runner.Run(t.Context(), agent, "start")
		require.NoError(t, err)

		assert.Equal(t, param.NewOpt("filtered-sync"), model.LastTurnArgs.SystemInstructions)
		require.IsType(t, agents.InputItems{}, model.LastTurnArgs.Input)
		input := model.LastTurnArgs.Input.(agents.InputItems)
		assert.Equal(t, "added-sync", input[len(input)-1].OfMessage.Content.OfString.Value)
	})

	t.Run("streamed", func(t *testing.T) {
		model := agentstesting.NewFakeModel(false, nil)
		agent := agents.New("test").WithModelInstance(model)

		// Prepare model output
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("ok"),
			},
		})

		filterFn := func(_ context.Context, data agents.CallModelData) (*agents.ModelInputData, error) {
			input := append(slices.Clone(data.ModelData.Input), agentstesting.GetTextInputItem("added-sync"))
			return &agents.ModelInputData{
				Input:        input,
				Instructions: param.NewOpt("filtered-sync"),
			}, nil
		}

		runner := agents.Runner{
			Config: agents.RunConfig{
				CallModelInputFilter: filterFn,
			},
		}
		result, err := runner.RunStreamed(t.Context(), agent, "start")
		require.NoError(t, err)
		err = result.StreamEvents(func(agents.StreamEvent) error { return nil })
		require.NoError(t, err)

		assert.Equal(t, param.NewOpt("filtered-sync"), model.LastTurnArgs.SystemInstructions)
		require.IsType(t, agents.InputItems{}, model.LastTurnArgs.Input)
		input := model.LastTurnArgs.Input.(agents.InputItems)
		assert.Equal(t, "added-sync", input[len(input)-1].OfMessage.Content.OfString.Value)
	})

	t.Run("error", func(t *testing.T) {
		model := agentstesting.NewFakeModel(false, nil)
		agent := agents.New("test").WithModelInstance(model)

		// Prepare model output
		model.SetNextOutput(agentstesting.FakeModelTurnOutput{
			Value: []agents.TResponseOutputItem{
				agentstesting.GetTextMessage("ok"),
			},
		})

		filterError := errors.New("filter error")
		filterFn := func(_ context.Context, data agents.CallModelData) (*agents.ModelInputData, error) {
			return nil, filterError
		}

		runner := agents.Runner{
			Config: agents.RunConfig{
				CallModelInputFilter: filterFn,
			},
		}
		_, err := runner.Run(t.Context(), agent, "start")
		require.ErrorIs(t, err, filterError)
	})
}
