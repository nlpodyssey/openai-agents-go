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
	"encoding/json"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPToolProxyConditionals(t *testing.T) {
	cond := &agents.MCPToolProxyExactNameConditionals{
		Names: []string{"ToolA", "ToolB"},
	}

	assert.True(t, cond.Check(&mcp.Tool{Name: "ToolA"}))
	assert.True(t, cond.Check(&mcp.Tool{Name: "ToolB"}))
	assert.False(t, cond.Check(&mcp.Tool{Name: "ToolC"}))
}

func TestMCPToolProxyForceResetInputParamsProcessor_NotFound(t *testing.T) {
	t.Run("NotFound_Error", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"missingParam": "someValue",
			},
			ResetNotFoundBehaviour: agents.MCPToolProxyResetNotFoundBehaviourError,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type:       "object",
				Properties: map[string]*jsonschema.Schema{},
			},
		}

		err := proc.ProcessSchema(tool)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missingParam")
	})

	t.Run("NotFound_Skip", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"missingParam": "someValue",
			},
			ResetNotFoundBehaviour: agents.MCPToolProxyResetNotFoundBehaviourSkip,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type:       "object",
				Properties: map[string]*jsonschema.Schema{},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)
		_, exists := tool.InputSchema.Properties["missingParam"]
		assert.False(t, exists)
	})

	t.Run("NotFound_Set", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"newString": "val",
				"newNumber": 42,
				"newBool":   true,
			},
			ResetNotFoundBehaviour:    agents.MCPToolProxyResetNotFoundBehaviourSet,
			ResetAdaptSchemaBehaviour: agents.MCPToolProxyResetAdaptSchemaBehaviourAnnotate,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type:       "object",
				Properties: map[string]*jsonschema.Schema{},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)

		propStr := tool.InputSchema.Properties["newString"]
		require.NotNil(t, propStr)
		assert.Equal(t, "string", propStr.Type)
		var strVal string
		err = json.Unmarshal(propStr.Default, &strVal)
		require.NoError(t, err)
		assert.Equal(t, "val", strVal)

		propNum := tool.InputSchema.Properties["newNumber"]
		require.NotNil(t, propNum)
		assert.Equal(t, "integer", propNum.Type)

		propBool := tool.InputSchema.Properties["newBool"]
		require.NotNil(t, propBool)
		assert.Equal(t, "boolean", propBool.Type)
	})
}

func TestMCPToolProxyForceResetInputParamsProcessor_NotMatchedTypes(t *testing.T) {
	t.Run("NotMatched_Error", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"param": 123, // integer
			},
			ResetNotMatchedTypesBehaviour: agents.MCPToolProxyResetNotMatchedTypesBehaviourError,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"param": {Type: "string"},
				},
			},
		}

		err := proc.ProcessSchema(tool)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "type mismatch")
	})

	t.Run("NotMatched_Skip", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"param": 123,
			},
			ResetNotMatchedTypesBehaviour: agents.MCPToolProxyResetNotMatchedTypesBehaviourSkip,
			ResetAdaptSchemaBehaviour:     agents.MCPToolProxyResetAdaptSchemaBehaviourAnnotate,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"param": {Type: "string"},
				},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)
		assert.Equal(t, "string", tool.InputSchema.Properties["param"].Type)
		assert.Nil(t, tool.InputSchema.Properties["param"].Default)
	})

	t.Run("NotMatched_Reset", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"param": 123,
			},
			ResetNotMatchedTypesBehaviour: agents.MCPToolProxyResetNotMatchedTypesBehaviourReset,
			ResetAdaptSchemaBehaviour:     agents.MCPToolProxyResetAdaptSchemaBehaviourAnnotate,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"param": {Type: "string"},
				},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)
		assert.Equal(t, "integer", tool.InputSchema.Properties["param"].Type)
		assert.NotNil(t, tool.InputSchema.Properties["param"].Default)
	})

	t.Run("Matched_NumberAndInteger", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"param": 123, // integer matches number in JSON Schema
			},
			ResetNotMatchedTypesBehaviour: agents.MCPToolProxyResetNotMatchedTypesBehaviourError,
			ResetAdaptSchemaBehaviour:     agents.MCPToolProxyResetAdaptSchemaBehaviourAnnotate,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"param": {Type: "number"},
				},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)
		assert.Equal(t, "number", tool.InputSchema.Properties["param"].Type)
		assert.NotNil(t, tool.InputSchema.Properties["param"].Default)
	})

	t.Run("Matched_BoolAndBoolean", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"param": true,
			},
			ResetNotMatchedTypesBehaviour: agents.MCPToolProxyResetNotMatchedTypesBehaviourError,
			ResetAdaptSchemaBehaviour:     agents.MCPToolProxyResetAdaptSchemaBehaviourAnnotate,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"param": {Type: "boolean"},
				},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)
		assert.Equal(t, "boolean", tool.InputSchema.Properties["param"].Type)
		assert.NotNil(t, tool.InputSchema.Properties["param"].Default)
	})
}

func TestMCPToolProxyForceResetInputParamsProcessor_AdaptSchemaBehaviour(t *testing.T) {
	t.Run("Hide_RemovesFromPropertiesAndRequired", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"secret": "hidden_val",
			},
			ResetAdaptSchemaBehaviour: agents.MCPToolProxyResetAdaptSchemaBehaviourHide,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"secret": {Type: "string", Description: "secret field"},
					"other":  {Type: "string"},
				},
				Required: []string{"secret", "other"},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)

		_, exists := tool.InputSchema.Properties["secret"]
		assert.False(t, exists)
		assert.NotContains(t, tool.InputSchema.Required, "secret")
		assert.Contains(t, tool.InputSchema.Required, "other")
	})

	t.Run("Default_ZeroValue_Is_Hide", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"secret": "hidden_val",
			},
			// ResetAdaptSchemaBehaviour is omitted (0)
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"secret": {Type: "string"},
				},
				Required: []string{"secret"},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)

		_, exists := tool.InputSchema.Properties["secret"]
		assert.False(t, exists)
		assert.NotContains(t, tool.InputSchema.Required, "secret")
	})

	t.Run("Keep_PreservesSchemaAsIs", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"param": "val",
			},
			ResetAdaptSchemaBehaviour: agents.MCPToolProxyResetAdaptSchemaBehaviourKeep,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"param": {Type: "string", Description: "original description"},
				},
				Required: []string{"param"},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)

		prop, exists := tool.InputSchema.Properties["param"]
		assert.True(t, exists)
		assert.Equal(t, "original description", prop.Description)
		assert.Nil(t, prop.Default)
		assert.Contains(t, tool.InputSchema.Required, "param")
	})

	t.Run("Annotate_SetsDefaultAndAnnotatesDescriptionAndRemovesRequired", func(t *testing.T) {
		proc := &agents.MCPToolProxyForceResetInputParamsProcessor{
			Params: map[string]any{
				"param": "admin",
			},
			ResetAdaptSchemaBehaviour: agents.MCPToolProxyResetAdaptSchemaBehaviourAnnotate,
		}

		tool := &mcp.Tool{
			Name: "testTool",
			InputSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"param": {Type: "string", Description: "user role"},
				},
				Required: []string{"param"},
			},
		}

		err := proc.ProcessSchema(tool)
		require.NoError(t, err)

		prop, exists := tool.InputSchema.Properties["param"]
		assert.True(t, exists)
		assert.NotNil(t, prop.Default)
		var defaultVal string
		err = json.Unmarshal(prop.Default, &defaultVal)
		require.NoError(t, err)
		assert.Equal(t, "admin", defaultVal)
		assert.Contains(t, prop.Description, "[preset: admin]")
		assert.NotContains(t, tool.InputSchema.Required, "param")
	})
}

func TestMCPToolProxyOrchestrator(t *testing.T) {
	orchestrator := agents.MCPToolProxyOrchestrator{
		Proxies: []agents.MCPToolProxy{
			{
				Conditionals: &agents.MCPToolProxyExactNameConditionals{
					Names: []string{"MyTool"},
				},
				Processors: []agents.MCPToolProxyProcessor{
					&agents.MCPToolProxyForceResetInputParamsProcessor{
						Params: map[string]any{
							"extra": "value",
						},
						ResetNotFoundBehaviour:    agents.MCPToolProxyResetNotFoundBehaviourSet,
						ResetAdaptSchemaBehaviour: agents.MCPToolProxyResetAdaptSchemaBehaviourAnnotate,
					},
				},
			},
		},
	}

	tool1 := &mcp.Tool{
		Name: "OtherTool",
		InputSchema: &jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{},
		},
	}
	tool2 := &mcp.Tool{
		Name: "MyTool",
		InputSchema: &jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{},
		},
	}

	err := orchestrator.Process([]*mcp.Tool{tool1, tool2})
	require.NoError(t, err)

	_, exists1 := tool1.InputSchema.Properties["extra"]
	assert.False(t, exists1)

	_, exists2 := tool2.InputSchema.Properties["extra"]
	assert.True(t, exists2)
}

func TestMCPToolProxyProcessArguments(t *testing.T) {
	orchestrator := agents.MCPToolProxyOrchestrator{
		Proxies: []agents.MCPToolProxy{
			{
				Conditionals: &agents.MCPToolProxyExactNameConditionals{
					Names: []string{"TargetTool"},
				},
				Processors: []agents.MCPToolProxyProcessor{
					&agents.MCPToolProxyRemoveInputParamsProcessor{
						Params: []string{"secretField", "unwanted"},
					},
					&agents.MCPToolProxyForceResetInputParamsProcessor{
						Params: map[string]any{
							"forced": "value123",
						},
					},
				},
			},
		},
	}

	t.Run("Matching tool removes and sets arguments", func(t *testing.T) {
		args := map[string]any{
			"normal":      "keepMe",
			"secretField": "removeMe",
			"unwanted":    "removeMeToo",
		}

		err := orchestrator.ProcessArguments("TargetTool", args)
		require.NoError(t, err)

		assert.Equal(t, "keepMe", args["normal"])
		assert.Equal(t, "value123", args["forced"])
		_, hasSecret := args["secretField"]
		assert.False(t, hasSecret)
		_, hasUnwanted := args["unwanted"]
		assert.False(t, hasUnwanted)
	})

	t.Run("Non-matching tool does not modify arguments", func(t *testing.T) {
		args := map[string]any{
			"normal":      "keepMe",
			"secretField": "removeMe",
		}

		err := orchestrator.ProcessArguments("OtherTool", args)
		require.NoError(t, err)

		assert.Equal(t, "keepMe", args["normal"])
		assert.Equal(t, "removeMe", args["secretField"])
		_, hasForced := args["forced"]
		assert.False(t, hasForced)
	})
}
