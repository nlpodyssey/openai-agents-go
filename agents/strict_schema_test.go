package agents_test

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmptySchemaHasAdditionalPropertiesFalse(t *testing.T) {
	schemas := []map[string]any{
		nil,
		{},
	}
	for _, schema := range schemas {
		t.Run(fmt.Sprintf("%#v", schema), func(t *testing.T) {
			strictSchema, err := agents.EnsureStrictJSONSchema(schema)
			require.NoError(t, err)
			assert.Contains(t, strictSchema, "additionalProperties")
			assert.Equal(t, false, strictSchema["additionalProperties"])
		})
	}
}

func TestObjectWithoutAdditionalProperties(t *testing.T) {
	// When an object type schema has properties but no additionalProperties,
	// it should be added and the "required" list set from the property keys.
	type m = map[string]any
	schema := m{
		"type": "object",
		"properties": m{
			"a": m{
				"type": "string",
			},
		},
	}
	result, err := agents.EnsureStrictJSONSchema(schema)
	require.NoError(t, err)
	assert.Equal(t, m{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"a"},
		"properties": m{
			"a": m{
				// The inner property remains unchanged
				// (no additionalProperties is added for non-object types)
				"type": "string",
			},
		},
	}, result)
}

func TestObjectWithTrueAdditionalProperties(t *testing.T) {
	// If additionalProperties is explicitly set to true for an object, a UserError should be raised.
	type m = map[string]any
	schema := m{
		"type": "object",
		"properties": m{
			"a": m{
				"type": "number",
			},
		},
		"additionalProperties": true,
	}
	_, err := agents.EnsureStrictJSONSchema(schema)
	assert.ErrorAs(t, err, &agents.UserError{})
}

func TestArrayItemsProcessingAndDefaultRemoval(t *testing.T) {
	// When processing an array, the items schema is processed recursively.
	// Also, any "default": nil should be removed.
	type m = map[string]any
	schema := m{
		"type": "array",
		"items": m{
			"type":    "number",
			"default": nil,
		},
	}
	result, err := agents.EnsureStrictJSONSchema(schema)
	require.NoError(t, err)
	assert.Equal(t, m{
		"type": "array",
		"items": m{
			"type": "number",
		},
	}, result)
}

func TestAnyOfProcessing(t *testing.T) {
	// Test that anyOf schemas are processed.
	type m = map[string]any
	schema := m{
		"anyOf": []any{
			m{"type": "object", "properties": m{"a": m{"type": "string"}}},
			m{"type": "number", "default": nil},
		},
	}
	result, err := agents.EnsureStrictJSONSchema(schema)
	require.NoError(t, err)
	assert.Equal(t, m{
		"anyOf": []any{
			m{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []any{"a"},
				"properties":           m{"a": m{"type": "string"}},
			},
			m{"type": "number"},
		},
	}, result)
}

func TestAllOfSingleEntryMerging(t *testing.T) {
	// When an allOf list has a single entry, its content should be merged into the parent.
	type m = map[string]any
	schema := m{
		"type": "object",
		"allOf": []any{
			m{"properties": m{"a": m{"type": "boolean"}}},
		},
	}
	result, err := agents.EnsureStrictJSONSchema(schema)
	require.NoError(t, err)
	assert.Equal(t, m{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"a"},
		"properties":           m{"a": m{"type": "boolean"}},
	}, result)
}

func TestDefaultRemovalOnNonObject(t *testing.T) {
	// Test that "default": nil is stripped from schemas that are not objects.
	type m = map[string]any
	schema := m{"type": "string", "default": nil}
	result, err := agents.EnsureStrictJSONSchema(schema)
	require.NoError(t, err)
	assert.Equal(t, m{"type": "string"}, result)
}

func TestRefExpansion(t *testing.T) {
	//Construct a schema with a definitions section and a property with a $ref.
	type m = map[string]any
	schema := m{
		"definitions": m{"refObj": m{"type": "string", "default": nil}},
		"type":        "object",
		"properties":  m{"a": m{"$ref": "#/definitions/refObj", "description": "desc"}},
	}
	result, err := agents.EnsureStrictJSONSchema(schema)
	require.NoError(t, err)
	assert.Equal(t, m{
		"definitions":          m{"refObj": m{"type": "string"}},
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"a"},
		"properties":           m{"a": m{"type": "string", "description": "desc"}},
	}, result)
}

func TestRefNoExpansionWhenAlone(t *testing.T) {
	// If the schema only contains a $ref key, it should not be expanded.
	type m = map[string]any
	schema := m{"$ref": "#/definitions/refObj"}
	result, err := agents.EnsureStrictJSONSchema(schema)
	require.NoError(t, err)
	assert.Equal(t, m{"$ref": "#/definitions/refObj"}, result)
}

func TestInvalidRefFormat(t *testing.T) {
	// A $ref that does not start with "#/" should trigger an error when resolved.
	type m = map[string]any
	schema := m{
		"type":       "object",
		"properties": m{"a": m{"$ref": "invalid", "description": "desc"}},
	}
	_, err := agents.EnsureStrictJSONSchema(schema)
	assert.Error(t, err)
}
