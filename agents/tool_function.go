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

package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/invopop/jsonschema"
	"github.com/nlpodyssey/openai-agents-go/util"
	"github.com/openai/openai-go/v3/packages/param"
)

// FunctionTool is a Tool that wraps a function.
type FunctionTool struct {
	// The name of the tool, as shown to the LLM. Generally the name of the function.
	Name string

	// A description of the tool, as shown to the LLM.
	Description string

	// The JSON schema for the tool's parameters.
	ParamsJSONSchema map[string]any

	// A function that invokes the tool with the given context and parameters.
	//
	// The params passed are:
	// 	1. The tool run context.
	// 	2. The arguments from the LLM, as a JSON string.
	//
	// You must return a string representation of the tool output.
	// In case of errors, you can either return an error (which will cause the run to fail) or
	// return a string error message (which will be sent back to the LLM).
	OnInvokeTool func(ctx context.Context, arguments string) (any, error)

	// Optional error handling function. When the tool invocation returns an error,
	// this function is called with the original error and its return value is sent
	// back to the LLM. If not set, a default function returning a generic error
	// message is used. To disable error handling and propagate the original error,
	// explicitly set this to a pointer to a nil ToolErrorFunction.
	FailureErrorFunction *ToolErrorFunction

	// Whether the JSON schema is in strict mode.
	// We **strongly** recommend setting this to True, as it increases the likelihood of correct JSON input.
	// Defaults to true if omitted.
	StrictJSONSchema param.Opt[bool]

	// Optional flag reporting whether the tool is enabled.
	// It can be either a boolean or a function which allows you to dynamically
	// enable/disable a tool based on your context/state.
	// Default value, if omitted: true.
	IsEnabled FunctionToolEnabler
}

func (t FunctionTool) ToolName() string {
	return t.Name
}

func (t FunctionTool) isTool() {}

// ToolErrorFunction is a callback that handles tool invocation errors and returns a value to be sent back to the LLM.
// If this function returns an error, it will be treated as a fatal error for the tool.
type ToolErrorFunction func(ctx context.Context, err error) (any, error)

// DefaultToolErrorFunction is the default handler used when a FunctionTool does not specify its own FailureErrorFunction.
// It returns a generic error message containing the original error string.
func DefaultToolErrorFunction(_ context.Context, err error) (any, error) {
	return fmt.Sprintf("An error occurred while running the tool. Please try again. Error: %s", err), nil
}

type FunctionToolEnabler interface {
	IsEnabled(ctx context.Context, agent *Agent) (bool, error)
}

// FunctionToolEnabledFlag is a static FunctionToolEnabler which always returns the configured flag value.
type FunctionToolEnabledFlag struct {
	isEnabled bool
}

func (f FunctionToolEnabledFlag) IsEnabled(context.Context, *Agent) (bool, error) {
	return f.isEnabled, nil
}

// NewFunctionToolEnabledFlag returns a FunctionToolEnabledFlag which always returns the configured flag value.
func NewFunctionToolEnabledFlag(isEnabled bool) FunctionToolEnabledFlag {
	return FunctionToolEnabledFlag{isEnabled: isEnabled}
}

// FunctionToolEnabled returns a static FunctionToolEnabler which always returns true.
func FunctionToolEnabled() FunctionToolEnabler {
	return NewFunctionToolEnabledFlag(true)
}

// FunctionToolDisabled returns a static FunctionToolEnabler which always returns false.
func FunctionToolDisabled() FunctionToolEnabler {
	return NewFunctionToolEnabledFlag(false)
}

// FunctionToolEnablerFunc can wrap a function to implement FunctionToolEnabler interface.
type FunctionToolEnablerFunc func(ctx context.Context, agent *Agent) (bool, error)

func (f FunctionToolEnablerFunc) IsEnabled(ctx context.Context, agent *Agent) (bool, error) {
	return f(ctx, agent)
}

// NewFunctionTool creates a FunctionTool tool with automatic JSON schema generation.
//
// This helper function simplifies tool creation by automatically generating the
// JSON schema from the Go types T (input arguments).
// The schema is generated using struct tags and Go reflection.
//
// It panics in case of errors. For a safer version, see SafeNewFunctionTool.
//
// Type parameters:
//   - T: The input argument type (must be JSON-serializable)
//   - R: The return value type
//
// Parameters:
//   - name: The tool name as shown to the LLM
//   - description: Optional tool description. If empty, no description is added
//   - handler: Function that processes the tool invocation
//
// The handler function receives:
//   - ctx: Context
//   - args: Parsed arguments of type T
//
// Schema generation behavior:
//   - Automatically reads and applies `jsonschema` struct tags for schema customization (e.g., `jsonschema:"enum=value1,enum=value2"`)
//   - Enables strict JSON schema mode by default
//
// Example:
//
//	type WeatherArgs struct {
//	    City string `json:"city"`
//	    Units string `json:"units" jsonschema:"enum=celsius,enum=fahrenheit"`
//	}
//
//	type WeatherResult struct {
//	    Temperature float64 `json:"temperature"`
//	    Conditions  string  `json:"conditions"`
//	}
//
//	func getWeather(ctx context.Context, args WeatherArgs) (WeatherResult, error) {
//	    // Implementation here
//	    return WeatherResult{Temperature: 22.5, Conditions: "sunny"}, nil
//	}
//
//	// Create tool with auto-generated schema
//	tool := NewFunctionTool("get_weather", "Get current weather", getWeather)
//
// For more control over the schema, create a FunctionTool manually instead.
func NewFunctionTool[T, R any](name string, description string, handler func(ctx context.Context, args T) (R, error)) FunctionTool {
	v, err := SafeNewFunctionTool(name, description, handler)
	if err != nil {
		panic(err)
	}
	return v
}

// SafeNewFunctionTool is like NewFunctionTool but returns an error instead of panicking.
func SafeNewFunctionTool[T, R any](name string, description string, handler func(ctx context.Context, args T) (R, error)) (FunctionTool, error) {
	reflector := &jsonschema.Reflector{
		ExpandedStruct:             true,
		RequiredFromJSONSchemaTags: false,
		AllowAdditionalProperties:  false,
	}

	var zero T
	var schema *jsonschema.Schema
	t := reflect.TypeOf(zero)
	if t.Kind() == reflect.Struct && t.Name() == "" && t.NumField() == 0 {
		// Avoid panic in jsonschema when reflecting an anonymous empty struct
		schema = &jsonschema.Schema{
			Version:    jsonschema.Version,
			Type:       "object",
			Properties: jsonschema.NewProperties(),
		}
		if !reflector.AllowAdditionalProperties {
			schema.AdditionalProperties = jsonschema.FalseSchema
		}
	} else {
		schema = reflector.Reflect(&zero)
	}

	schemaMap, err := util.JSONMap(schema)
	if err != nil {
		return FunctionTool{}, fmt.Errorf("failed to transform function tool jsonschema.Schema to map: %w", err)
	}

	schemaMap, err = EnsureStrictJSONSchema(schemaMap)
	if err != nil {
		return FunctionTool{}, fmt.Errorf("failed to ensure strictness of function tool json schema: %w", err)
	}

	// Add description at the top level if provided
	if description != "" && schemaMap != nil {
		schemaMap["description"] = description
	}

	return FunctionTool{
		Name:             name,
		ParamsJSONSchema: schemaMap,
		StrictJSONSchema: param.NewOpt(true),
		OnInvokeTool: func(ctx context.Context, arguments string) (any, error) {
			var args T
			if err := json.Unmarshal([]byte(arguments), &args); err != nil {
				return nil, fmt.Errorf("failed to parse arguments: %w", err)
			}
			return handler(ctx, args)
		},
	}, nil
}
