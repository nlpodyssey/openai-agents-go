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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOutputType(t *testing.T) {
	type Foo struct {
		Bar string `json:"bar"`
	}

	type m = map[string]any

	t.Run("structured output struct", func(t *testing.T) {
		ot := agents.OutputType[Foo]()

		assert.False(t, ot.IsPlainText())
		assert.Equal(t, "agents_test.Foo", ot.Name())
		assert.True(t, ot.IsStrictJSONSchema())

		schema, err := ot.JSONSchema()
		require.NoError(t, err)
		assert.Equal(t, m{
			"$schema":              "https://json-schema.org/draft/2020-12/schema",
			"type":                 "object",
			"required":             []string{"bar"},
			"additionalProperties": false,
			"properties":           m{"bar": m{"type": "string"}},
		}, schema)

		validated, err := ot.ValidateJSON(t.Context(), `{"bar": "baz"}`)
		require.NoError(t, err)
		assert.Equal(t, Foo{Bar: "baz"}, validated)
	})

	t.Run("structured output list", func(t *testing.T) {
		ot := agents.OutputType[[]string]()

		assert.False(t, ot.IsPlainText())
		assert.Equal(t, "[]string", ot.Name())
		assert.True(t, ot.IsStrictJSONSchema())

		schema, err := ot.JSONSchema()
		require.NoError(t, err)
		assert.Equal(t, m{
			"$schema":              "https://json-schema.org/draft/2020-12/schema",
			"type":                 "object",
			"required":             []string{"response"},
			"additionalProperties": false,
			"properties":           m{"response": m{"type": "array", "items": m{"type": "string"}}},
		}, schema)

		validated, err := ot.ValidateJSON(t.Context(), `{"response": ["foo", "bar"]}`)
		require.NoError(t, err)
		assert.Equal(t, []string{"foo", "bar"}, validated)
	})

	t.Run("bad JSON causes error", func(t *testing.T) {
		ot := agents.OutputType[Foo]()

		badValues := []string{
			`not valid JSON`,
			`["foo"]`,
			`{}`,                       // required "bar" is missing
			`{"value": "foo"}`,         // required "bar" is missing + no additional props
			`{"bar": "baz", "x": "y"}`, // no additional props
		}
		for _, val := range badValues {
			_, err := ot.ValidateJSON(t.Context(), val)
			require.ErrorAs(t, err, &agents.ModelBehaviorError{})
		}
	})

	t.Run("plain text output type does not produce schema", func(t *testing.T) {
		ot := agents.OutputType[string]()

		assert.True(t, ot.IsPlainText())
		assert.Equal(t, "string", ot.Name())
		assert.True(t, ot.IsStrictJSONSchema())

		_, err := ot.JSONSchema()
		require.ErrorAs(t, err, &agents.UserError{})

		_, err = ot.ValidateJSON(t.Context(), `"foo"`)
		require.ErrorAs(t, err, &agents.UserError{})
	})

	t.Run("option non-strict schema", func(t *testing.T) {
		ot := agents.OutputTypeWithOpts[Foo](agents.OutputTypeOpts{
			StrictJSONSchema: false,
		})

		assert.False(t, ot.IsPlainText())
		assert.Equal(t, "agents_test.Foo", ot.Name())
		assert.False(t, ot.IsStrictJSONSchema())

		schema, err := ot.JSONSchema()
		require.NoError(t, err)
		assert.Equal(t, m{
			"$schema":    "https://json-schema.org/draft/2020-12/schema",
			"type":       "object",
			"required":   []any{"bar"},
			"properties": m{"bar": m{"type": "string"}},
		}, schema)

		validated, err := ot.ValidateJSON(t.Context(), `{"bar": "baz", "x": "y"}`)
		require.NoError(t, err)
		assert.Equal(t, Foo{Bar: "baz"}, validated)
	})
}

var CustomOutputTypeJSONSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"foo": map[string]any{"type": "string"},
	},
	"required": []string{"foo"},
}

type CustomOutputType struct{}

func (CustomOutputType) IsPlainText() bool                   { return false }
func (CustomOutputType) Name() string                        { return "FooBarBaz" }
func (CustomOutputType) IsStrictJSONSchema() bool            { return false }
func (CustomOutputType) JSONSchema() (map[string]any, error) { return CustomOutputTypeJSONSchema, nil }
func (CustomOutputType) ValidateJSON(context.Context, string) (any, error) {
	return []string{"some", "output"}, nil
}

func TestCustomOutputType(t *testing.T) {
	ot := CustomOutputType{}

	assert.False(t, ot.IsPlainText())
	assert.Equal(t, "FooBarBaz", ot.Name())
	assert.False(t, ot.IsStrictJSONSchema())

	schema, err := ot.JSONSchema()
	require.NoError(t, err)
	assert.Equal(t, CustomOutputTypeJSONSchema, schema)

	validated, err := ot.ValidateJSON(t.Context(), `{"foo": "bar"}`)
	require.NoError(t, err)
	assert.Equal(t, []string{"some", "output"}, validated)
}
