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
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/matteo-grella/dwarfreflect"
	"github.com/nlpodyssey/openai-agents-go/util/transforms"
	"github.com/openai/openai-go/v3/packages/param"
)

// resultExtractor defines how to extract (any, error) from function call results
type resultExtractor func(results []reflect.Value) (any, error)

// NewFunctionToolAny creates a FunctionTool from any function signature.
// Requires DWARF debug info (go build default). Returns an error if stripped with -ldflags="-w" or go run.
//
// This function automatically extracts parameter names and types from any function,
// creates a dynamic struct for arguments, and generates JSON schema automatically.
//
// Context handling:
//   - context.Context is optional and can be in any position
//   - context.Context parameters are automatically detected and excluded from JSON schema
//   - Context is injected automatically during function calls
//
// Naming conventions (controlled by OPENAI_AGENTS_NAMING_CONVENTION environment variable):
//   - "snake_case" (default): function names and JSON tags use snake_case
//   - "camelCase": function names and JSON tags use camelCase
//
// Parameters:
//   - name: The tool name as shown to the LLM. If empty (""), automatically deduced from function name.
//   - description: Optional tool description
//   - handler: Any function (context.Context is optional)
func NewFunctionToolAny(name string, description string, handler any) (FunctionTool, error) {
	tool, err := dwarfreflect.NewFunction(handler)
	if err != nil {
		return FunctionTool{}, err
	}

	if name == "" {
		name = transforms.ToCase(tool.GetBaseFunctionName())
	}

	structOpts := createStructOptions()
	argStructType := tool.GetNonContextStructTypeWithOptions(structOpts)
	argParamNames, _ := tool.GetNonContextParameters()
	extractResult := createResultExtractor(tool)

	// Generate JSON schema
	var schemaMap map[string]any
	if len(argParamNames) == 0 {
		// No parameters besides context(s)
		schemaMap = map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	} else {
		schemaMap = generateJSONSchema(argStructType, tool.GetBaseFunctionName(), description)
	}

	return FunctionTool{
		Name:             name,
		ParamsJSONSchema: schemaMap,
		StrictJSONSchema: param.NewOpt(true),
		OnInvokeTool: func(ctx context.Context, arguments string) (any, error) {
			if len(argParamNames) == 0 {
				results, err := tool.CallWithContext(ctx)
				if err != nil {
					return nil, err
				}
				return extractResult(results)
			}

			// Parse JSON arguments into dynamic struct
			argStructPtr := reflect.New(argStructType)
			if err := json.Unmarshal([]byte(arguments), argStructPtr.Interface()); err != nil {
				return nil, fmt.Errorf("failed to parse arguments: %w", err)
			}

			results, err := tool.CallWithNonContextStructAndContext(ctx, argStructPtr.Elem().Interface())
			if err != nil {
				return nil, err
			}
			return extractResult(results)
		},
	}, nil
}

// createStructOptions creates StructOptions for JSON schema generation
func createStructOptions() dwarfreflect.StructOptions {
	return dwarfreflect.StructOptions{
		FieldNamer: func(paramName string) string {
			// Capitalize first letter for exported fields
			return strings.Title(paramName)
		},
		TagBuilder: func(paramName string, paramType reflect.Type) string {
			jsonName := transforms.ToCase(paramName)
			tag := fmt.Sprintf(`json:"%s"`, jsonName)

			if constraints := getTypeConstraints(paramType); constraints != "" {
				tag += fmt.Sprintf(` jsonschema:"%s"`, constraints)
			}

			return tag
		},
	}
}

// generateJSONSchema creates JSON schema from the dynamic struct type
func generateJSONSchema(structType reflect.Type, functionName string, description string) map[string]any {
	reflector := &jsonschema.Reflector{
		ExpandedStruct:             true,
		RequiredFromJSONSchemaTags: false,
		AllowAdditionalProperties:  false,
	}

	reflector.Namer = func(t reflect.Type) string {
		if t == structType {
			return functionName + "Params"
		}
		return t.Name()
	}

	schema := reflector.ReflectFromType(structType)

	schemaBytes, _ := json.Marshal(schema)
	var schemaMap map[string]any
	json.Unmarshal(schemaBytes, &schemaMap)

	// Add description at the top level if provided
	if description != "" && schemaMap != nil {
		schemaMap["description"] = description
	}

	return schemaMap
}

// createResultExtractor analyzes function signature to create appropriate result extractor
func createResultExtractor(tool *dwarfreflect.Function) resultExtractor {
	returnTypes, lastIsError := tool.GetReturnInfo()
	numReturns := len(returnTypes)

	return func(results []reflect.Value) (any, error) {
		switch numReturns {
		case 0:
			return nil, nil
		case 1:
			result := results[0].Interface()
			if lastIsError {
				// Function signature indicates this should be treated as error
				if err, ok := result.(error); ok {
					return nil, err
				}
			}
			return result, nil
		case 2:
			if lastIsError {
				// Standard (result, error) pattern
				result := results[0].Interface()
				if errVal := results[1]; !errVal.IsNil() {
					if err, ok := errVal.Interface().(error); ok {
						return result, err
					}
				}
				return result, nil
			}
			// Two results, neither is error type - return as tuple
			return []any{results[0].Interface(), results[1].Interface()}, nil
		default:
			// Multiple return values
			var resultSlice []any
			for i, result := range results {
				if i == len(results)-1 && lastIsError {
					// Last value is error according to function signature
					if errVal := result; !errVal.IsNil() {
						if err, ok := errVal.Interface().(error); ok {
							if len(resultSlice) == 1 {
								return resultSlice[0], err
							}
							return resultSlice, err
						}
					}
				}
				resultSlice = append(resultSlice, result.Interface())
			}
			return resultSlice, nil
		}
	}
}

// getTypeConstraints adds JSON schema constraints based on Go types
func getTypeConstraints(paramType reflect.Type) string {
	switch paramType.Kind() {
	case reflect.String:
		return ""
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return "type=integer"
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "type=integer,minimum=0"
	case reflect.Float32, reflect.Float64:
		return "type=number"
	case reflect.Bool:
		return "type=boolean"
	case reflect.Slice:
		return "type=array"
	case reflect.Map:
		return "type=object"
	default:
		return ""
	}
}
