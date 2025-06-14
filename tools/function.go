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

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared/constant"
)

// Function Tool that wraps a function.
type Function struct {
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

func (f Function) ToolName() string {
	return f.Name
}

func (f Function) ConvertToResponses(context.Context) (*responses.ToolUnionParam, *responses.ResponseIncludable, error) {
	return &responses.ToolUnionParam{
		OfFunction: &responses.FunctionToolParam{
			Name:        f.Name,
			Parameters:  f.ParamsJSONSchema,
			Strict:      param.NewOpt(f.StrictJSONSchema.Or(true)),
			Description: param.NewOpt(f.Description),
			Type:        constant.ValueOf[constant.Function](),
		},
	}, nil, nil
}

func (f Function) ConvertToChatCompletions(context.Context) (*openai.ChatCompletionToolParam, error) {
	description := param.Null[string]()
	if f.Description != "" {
		description = param.NewOpt(f.Description)
	}
	return &openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        f.Name,
			Description: description,
			Parameters:  f.ParamsJSONSchema,
		},
		Type: constant.ValueOf[constant.Function](),
	}, nil
}

type FunctionToolEnabler interface {
	IsEnabled(ctx context.Context) (bool, error)
	// TODO: should also have Agent param, but for now it would imply a circular package dependency
}

// FunctionToolEnabledFlag is a static FunctionToolEnabler which always returns the configured flag value.
type FunctionToolEnabledFlag struct {
	isEnabled bool
}

func (f FunctionToolEnabledFlag) IsEnabled(context.Context) (bool, error) {
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
type FunctionToolEnablerFunc func(ctx context.Context) (bool, error)

func (f FunctionToolEnablerFunc) IsEnabled(ctx context.Context) (bool, error) {
	return f(ctx)
}

// NewFunctionTool creates a Function tool with automatic JSON schema generation.
//
// This helper function simplifies tool creation by automatically generating the
// JSON schema from the Go types T (input arguments).
// The schema is generated using struct tags and Go reflection.
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
// For more control over the schema, create a Function manually instead.
func NewFunctionTool[T, R any](name string, description string, handler func(ctx context.Context, args T) (R, error)) Function {
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

	schemaBytes, _ := json.Marshal(schema)
	var schemaMap map[string]any
	json.Unmarshal(schemaBytes, &schemaMap)

	// Add description at the top level if provided
	if description != "" && schemaMap != nil {
		schemaMap["description"] = description
	}

	return Function{
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
	}
}
