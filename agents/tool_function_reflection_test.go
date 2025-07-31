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
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

// Test functions for various signatures
func simpleFunction(name string, age int) string {
	return name + " is " + string(rune(age)) + " years old"
}

func functionWithContext(ctx context.Context, message string) string {
	return "processed: " + message
}

func functionNoParams() string {
	return "no params"
}

func functionWithError(input string) (string, error) {
	if input == "error" {
		return "", errors.New("test error")
	}
	return "success: " + input, nil
}

func functionMultipleReturns(a, b int) (int, int) {
	return a + b, a * b
}

func functionOnlyError() error {
	return errors.New("only error")
}

func functionWithSlice(items []string) int {
	return len(items)
}

func functionWithMap(data map[string]int) int {
	sum := 0
	for _, v := range data {
		sum += v
	}
	return sum
}

func functionWithFloat(price float64) float64 {
	return price * 1.1
}

func functionWithBool(flag bool) string {
	if flag {
		return "true"
	}
	return "false"
}

func functionWithUint(count uint32) uint32 {
	return count + 1
}

func mustNewFunctionTool(t *testing.T, name string, description string, fn any) FunctionTool {
	t.Helper()
	f, err := NewFunctionToolAny(name, description, fn)
	if err != nil {
		if strings.Contains(err.Error(), "DWARF") {
			t.Skipf("DWARF not available: %v", err)
		}
		t.Fatalf("unexpected error: %v", err)
	}
	return f
}

func TestNewFunctionToolAny_BasicFunction(t *testing.T) {
	tool := mustNewFunctionTool(t, "", "Simple string function", simpleFunction)

	// Check that the name was transformed according to the current convention
	// We don't know which convention is set, so we check both possibilities
	validNames := map[string]bool{
		"simple_function": true, // snake_case
		"simpleFunction":  true, // camelCase
	}

	if !validNames[tool.Name] {
		t.Errorf("Expected tool name to be either 'simple_function' or 'simpleFunction', got '%s'", tool.Name)
	}

	// Check JSON schema
	schemaJSON, err := json.Marshal(tool.ParamsJSONSchema)
	if err != nil {
		t.Fatalf("Failed to marshal schema: %v", err)
	}

	var schema map[string]any
	if err := json.Unmarshal(schemaJSON, &schema); err != nil {
		t.Fatalf("Failed to unmarshal schema: %v", err)
	}

	// Verify schema structure
	if schema["type"] != "object" {
		t.Errorf("Expected schema type 'object', got '%v'", schema["type"])
	}

	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("Schema properties not found or wrong type")
	}

	// Check 'name' parameter
	nameParam, ok := properties["name"].(map[string]any)
	if !ok {
		t.Fatal("Name parameter not found in schema")
	}
	if nameParam["type"] != "string" {
		t.Errorf("Expected name type 'string', got '%v'", nameParam["type"])
	}

	// Check 'age' parameter
	ageParam, ok := properties["age"].(map[string]any)
	if !ok {
		t.Fatal("Age parameter not found in schema")
	}
	if ageParam["type"] != "integer" {
		t.Errorf("Expected age type 'integer', got '%v'", ageParam["type"])
	}
}

func TestNewFunctionToolAny_WithContext(t *testing.T) {
	tool := mustNewFunctionTool(t, "process_message", "Function with context", functionWithContext)

	// Test invocation
	ctx := context.Background()
	result, err := tool.OnInvokeTool(ctx, `{"message": "hello"}`)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if result != "processed: hello" {
		t.Errorf("Expected 'processed: hello', got '%v'", result)
	}

	// Verify context is not in schema
	properties, ok := tool.ParamsJSONSchema["properties"].(map[string]any)
	if !ok {
		t.Fatal("Schema properties not found")
	}

	if _, hasContext := properties["ctx"]; hasContext {
		t.Error("Context parameter should not be in schema")
	}

	if _, hasMessage := properties["message"]; !hasMessage {
		t.Error("Message parameter should be in schema")
	}
}

func TestNewFunctionToolAny_NoParameters(t *testing.T) {
	tool := mustNewFunctionTool(t, "", "", functionNoParams)

	// Check schema for no parameters
	properties, ok := tool.ParamsJSONSchema["properties"].(map[string]any)
	if !ok {
		t.Fatal("Schema properties not found")
	}

	if len(properties) != 0 {
		t.Errorf("Expected no properties, got %d", len(properties))
	}

	// Test invocation
	ctx := context.Background()
	result, err := tool.OnInvokeTool(ctx, "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if result != "no params" {
		t.Errorf("Expected 'no params', got '%v'", result)
	}
}

func TestNewFunctionToolAny_WithError(t *testing.T) {
	tool := mustNewFunctionTool(t, "", "", functionWithError)

	ctx := context.Background()

	// Test successful case
	result, err := tool.OnInvokeTool(ctx, `{"input": "test"}`)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result != "success: test" {
		t.Errorf("Expected 'success: test', got '%v'", result)
	}

	// Test error case
	result, err = tool.OnInvokeTool(ctx, `{"input": "error"}`)
	if err == nil {
		t.Fatal("Expected error, got none")
	}
	if err.Error() != "test error" {
		t.Errorf("Expected 'test error', got '%v'", err.Error())
	}
}

func TestNewFunctionToolAny_MultipleReturns(t *testing.T) {
	tool := mustNewFunctionTool(t, "", "", functionMultipleReturns)

	ctx := context.Background()
	result, err := tool.OnInvokeTool(ctx, `{"a": 3, "b": 4}`)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Result should be a slice with two values
	resultSlice, ok := result.([]any)
	if !ok {
		t.Fatalf("Expected result to be []any, got %T", result)
	}

	if len(resultSlice) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(resultSlice))
	}

	if resultSlice[0] != 7 { // 3 + 4
		t.Errorf("Expected first result 7, got %v", resultSlice[0])
	}
	if resultSlice[1] != 12 { // 3 * 4
		t.Errorf("Expected second result 12, got %v", resultSlice[1])
	}
}

func TestNewFunctionToolAny_OnlyError(t *testing.T) {
	tool := mustNewFunctionTool(t, "", "", functionOnlyError)

	ctx := context.Background()
	result, err := tool.OnInvokeTool(ctx, "")
	if err == nil {
		t.Fatal("Expected error, got none")
	}
	if err.Error() != "only error" {
		t.Errorf("Expected 'only error', got '%v'", err.Error())
	}
	if result != nil {
		t.Errorf("Expected nil result, got %v", result)
	}
}

func TestNewFunctionToolAny_TypeConstraints(t *testing.T) {
	tests := []struct {
		name                string
		function            any
		paramName           string
		expectedType        string
		expectedConstraints map[string]any
	}{
		{
			name:         "slice parameter",
			function:     functionWithSlice,
			paramName:    "items",
			expectedType: "array",
		},
		{
			name:         "map parameter",
			function:     functionWithMap,
			paramName:    "data",
			expectedType: "object",
		},
		{
			name:         "float parameter",
			function:     functionWithFloat,
			paramName:    "price",
			expectedType: "number",
		},
		{
			name:         "bool parameter",
			function:     functionWithBool,
			paramName:    "flag",
			expectedType: "boolean",
		},
		{
			name:                "uint parameter",
			function:            functionWithUint,
			paramName:           "count",
			expectedType:        "integer",
			expectedConstraints: map[string]any{"minimum": float64(0)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := mustNewFunctionTool(t, "", "", tt.function)

			properties, ok := tool.ParamsJSONSchema["properties"].(map[string]any)
			if !ok {
				t.Fatal("Schema properties not found")
			}

			param, ok := properties[tt.paramName].(map[string]any)
			if !ok {
				t.Fatalf("Parameter '%s' not found in schema", tt.paramName)
			}

			if param["type"] != tt.expectedType {
				t.Errorf("Expected type '%s', got '%v'", tt.expectedType, param["type"])
			}

			// Check additional constraints
			for constraint, expectedValue := range tt.expectedConstraints {
				if param[constraint] != expectedValue {
					t.Errorf("Expected %s '%v', got '%v'", constraint, expectedValue, param[constraint])
				}
			}
		})
	}
}

func TestNewFunctionToolAny_CustomName(t *testing.T) {
	tool := mustNewFunctionTool(t, "custom_tool_name", "Custom description", simpleFunction)

	if tool.Name != "custom_tool_name" {
		t.Errorf("Expected tool name 'custom_tool_name', got '%s'", tool.Name)
	}
}

func TestNewFunctionToolAny_InvalidJSON(t *testing.T) {
	tool := mustNewFunctionTool(t, "", "", simpleFunction)

	ctx := context.Background()
	_, err := tool.OnInvokeTool(ctx, `{"name": "test", "age": "not a number"}`)
	if err == nil {
		t.Fatal("Expected error for invalid JSON, got none")
	}
}

func TestGetTypeConstraints(t *testing.T) {
	tests := []struct {
		name     string
		typ      reflect.Type
		expected string
	}{
		{"string", reflect.TypeOf(""), ""},
		{"int", reflect.TypeOf(0), "type=integer"},
		{"uint", reflect.TypeOf(uint(0)), "type=integer,minimum=0"},
		{"float64", reflect.TypeOf(0.0), "type=number"},
		{"bool", reflect.TypeOf(false), "type=boolean"},
		{"slice", reflect.TypeOf([]string{}), "type=array"},
		{"map", reflect.TypeOf(map[string]int{}), "type=object"},
		{"struct", reflect.TypeOf(struct{}{}), ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getTypeConstraints(tt.typ)
			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

// Test naming convention behavior
func TestNamingConventionIntegration(t *testing.T) {
	// Note: Since the naming convention is loaded at init time,
	// we can only test against what's currently set

	tool := mustNewFunctionTool(t, "", "", simpleFunction)

	// Get the current convention from environment
	convention := os.Getenv("OPENAI_AGENTS_NAMING_CONVENTION")

	switch convention {
	case "camelCase":
		if tool.Name != "simpleFunction" {
			t.Errorf("Expected camelCase name 'simpleFunction', got '%s'", tool.Name)
		}
	case "snake_case", "": // empty means default (snake_case)
		if tool.Name != "simple_function" {
			t.Errorf("Expected snake_case name 'simple_function', got '%s'", tool.Name)
		}
	default:
		t.Errorf("Unknown naming convention: %s", convention)
	}

	// Also verify the JSON field names match the convention
	properties, _ := tool.ParamsJSONSchema["properties"].(map[string]any)

	// Check for field name presence based on convention
	switch convention {
	case "camelCase":
		if _, hasName := properties["name"]; !hasName {
			t.Error("Expected 'name' field in camelCase schema")
		}
		if _, hasAge := properties["age"]; !hasAge {
			t.Error("Expected 'age' field in camelCase schema")
		}
	case "snake_case", "":
		if _, hasName := properties["name"]; !hasName {
			t.Error("Expected 'name' field in snake_case schema")
		}
		if _, hasAge := properties["age"]; !hasAge {
			t.Error("Expected 'age' field in snake_case schema")
		}
	}
}

// Complex function for edge case testing
func complexFunction(ctx context.Context, id int, data map[string][]float64, options struct{ Enabled bool }) (map[string]any, error) {
	if !options.Enabled {
		return nil, errors.New("disabled")
	}
	result := make(map[string]any)
	result["id"] = id
	result["count"] = len(data)
	return result, nil
}

func TestNewFunctionToolAny_ComplexFunction(t *testing.T) {
	tool := mustNewFunctionTool(t, "", "Complex function test", complexFunction)

	// Verify schema has correct parameters (excluding context)
	properties, ok := tool.ParamsJSONSchema["properties"].(map[string]any)
	if !ok {
		t.Fatal("Schema properties not found")
	}

	// Should have id, data, and options
	if _, hasID := properties["id"]; !hasID {
		t.Error("Missing 'id' parameter in schema")
	}
	if _, hasData := properties["data"]; !hasData {
		t.Error("Missing 'data' parameter in schema")
	}
	if _, hasOptions := properties["options"]; !hasOptions {
		t.Error("Missing 'options' parameter in schema")
	}
	if _, hasContext := properties["ctx"]; hasContext {
		t.Error("Context should not be in schema")
	}

	// Test successful invocation
	ctx := context.Background()
	jsonArgs := `{
		"id": 123,
		"data": {"series1": [1.0, 2.0, 3.0], "series2": [4.0, 5.0]},
		"options": {"Enabled": true}
	}`

	result, err := tool.OnInvokeTool(ctx, jsonArgs)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	resultMap, ok := result.(map[string]any)
	if !ok {
		t.Fatalf("Expected map result, got %T", result)
	}

	if resultMap["id"] != 123 {
		t.Errorf("Expected id 123, got %v", resultMap["id"])
	}
	if resultMap["count"] != 2 {
		t.Errorf("Expected count 2, got %v", resultMap["count"])
	}
}
