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
	"context"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/agentstesting"
	"github.com/nlpodyssey/openai-agents-go/usage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunAddsUsageToExistingContext(t *testing.T) {
	model := agentstesting.NewFakeModel(false, &agentstesting.FakeModelTurnOutput{
		Value: []agents.TResponseOutputItem{agentstesting.GetTextMessage("hello")},
	})
	model.SetHardcodedUsage(usage.Usage{InputTokens: 5, OutputTokens: 3, TotalTokens: 8})

	agent := agents.New("test").WithModelInstance(model)

	tracker := usage.NewUsage()
	ctx := usage.NewContext(context.Background(), tracker)

	_, err := agents.Run(ctx, agent, "hi")
	require.NoError(t, err)

	assert.Equal(t, uint64(1), tracker.Requests)
	assert.Equal(t, uint64(5), tracker.InputTokens)
	assert.Equal(t, uint64(3), tracker.OutputTokens)
	assert.Equal(t, uint64(8), tracker.TotalTokens)
}
