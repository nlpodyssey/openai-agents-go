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
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPToolProxyOrchestrator contains any proxies
type MCPToolProxyOrchestrator struct {
	Proxies []MCPToolProxy
}

// Process slice of MCP tools through any proxies
func (o *MCPToolProxyOrchestrator) Process(tools []*mcp.Tool) error {
	for _, tool := range tools {
		for _, proxy := range o.Proxies {
			if proxy.CheckConditionals(tool) {
				err := proxy.Process(tool)
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// ProcessArguments passes tool arguments through matching proxies
func (o *MCPToolProxyOrchestrator) ProcessArguments(toolName string, arguments map[string]any) error {
	tool := &mcp.Tool{Name: toolName}
	for _, proxy := range o.Proxies {
		if proxy.CheckConditionals(tool) {
			err := proxy.ProcessArguments(arguments)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// MCPToolProxy contains conditionals and algorithm of passing tool through proxy
type MCPToolProxy struct {
	Conditionals MCPToolProxyConditionals
	Processors   []MCPToolProxyProcessor
}

// CheckConditionals returns true if tool will be passed through proxy
func (p *MCPToolProxy) CheckConditionals(tool *mcp.Tool) bool {
	return p.Conditionals.Check(tool)
}

// Process MCP tool through proxy and return the new one
func (p *MCPToolProxy) Process(tool *mcp.Tool) error {
	for _, processor := range p.Processors {
		err := processor.ProcessSchema(tool)
		if err != nil {
			return err
		}
	}
	return nil
}

// ProcessArguments passes tool arguments through any processors
func (p *MCPToolProxy) ProcessArguments(arguments map[string]any) error {
	for _, processor := range p.Processors {
		err := processor.ProcessArguments(arguments)
		if err != nil {
			return err
		}
	}
	return nil
}

type MCPToolProxyConditionals interface {
	Check(tool *mcp.Tool) bool
}

type MCPToolProxyExactNameConditionals struct {
	Names []string
}

func (c *MCPToolProxyExactNameConditionals) Check(tool *mcp.Tool) bool {
	return slices.Contains(c.Names, tool.Name)
}

type MCPToolProxyProcessor interface {
	ProcessSchema(tool *mcp.Tool) error
	ProcessArguments(arguments map[string]any) error
}

type MCPToolProxyRemoveRequiredBehaviour int

const (
	MCPToolProxyRemoveRequiredBehaviourError MCPToolProxyRemoveRequiredBehaviour = iota
	MCPToolProxyRemoveRequiredBehaviourSkip
	MCPToolProxyRemoveRequiredBehaviourRemove
)

func proxyInputSchema(tool *mcp.Tool) (*jsonschema.Schema, error) {
	if schema, ok := tool.InputSchema.(*jsonschema.Schema); ok && schema != nil {
		return schema, nil
	}
	if tool.InputSchema == nil {
		schema := &jsonschema.Schema{Type: "object"}
		tool.InputSchema = schema
		return schema, nil
	}
	data, err := json.Marshal(tool.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("marshal input schema: %w", err)
	}
	var schema jsonschema.Schema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("decode input schema: %w", err)
	}
	tool.InputSchema = &schema
	return &schema, nil
}

type MCPToolProxyRemoveInputParamsProcessor struct {
	Params                  []string
	RemoveRequiredBehaviour MCPToolProxyRemoveRequiredBehaviour
}

func (p *MCPToolProxyRemoveInputParamsProcessor) ProcessSchema(tool *mcp.Tool) error {
	if tool == nil || tool.InputSchema == nil {
		return nil
	}

	schema, err := proxyInputSchema(tool)
	if err != nil {
		return err
	}

	for _, paramName := range p.Params {
		isRequired := slices.Contains(schema.Required, paramName)
		if isRequired {
			switch p.RemoveRequiredBehaviour {
			case MCPToolProxyRemoveRequiredBehaviourError:
				return fmt.Errorf("param is required: %s", paramName) // TODO: custom error
			case MCPToolProxyRemoveRequiredBehaviourSkip:
				continue
			case MCPToolProxyRemoveRequiredBehaviourRemove:
				delete(schema.Properties, paramName)
				schema.Required = slices.DeleteFunc(schema.Required, func(s string) bool {
					return s == paramName
				})
			}
		} else {
			delete(schema.Properties, paramName)
		}
	}

	return nil
}

func (p *MCPToolProxyRemoveInputParamsProcessor) ProcessArguments(arguments map[string]any) error {
	for _, paramName := range p.Params {
		delete(arguments, paramName)
	}
	return nil
}

type MCPToolProxyResetNotFoundBehaviour int

const (
	MCPToolProxyResetNotFoundBehaviourError MCPToolProxyResetNotFoundBehaviour = iota
	MCPToolProxyResetNotFoundBehaviourSkip
	MCPToolProxyResetNotFoundBehaviourSet
)

type MCPToolProxyResetNotMatchedTypesBehaviour int

const (
	MCPToolProxyResetNotMatchedTypesBehaviourReset MCPToolProxyResetNotMatchedTypesBehaviour = iota
	MCPToolProxyResetNotMatchedTypesBehaviourSkip
	MCPToolProxyResetNotMatchedTypesBehaviourError
)

type MCPToolProxyResetAdaptSchemaBehaviour int

const (
	MCPToolProxyResetAdaptSchemaBehaviourHide MCPToolProxyResetAdaptSchemaBehaviour = iota
	MCPToolProxyResetAdaptSchemaBehaviourKeep
	MCPToolProxyResetAdaptSchemaBehaviourAnnotate
)

type MCPToolProxyForceResetInputParamsProcessor struct {
	Params                        map[string]any
	ResetNotFoundBehaviour        MCPToolProxyResetNotFoundBehaviour
	ResetNotMatchedTypesBehaviour MCPToolProxyResetNotMatchedTypesBehaviour
	ResetAdaptSchemaBehaviour     MCPToolProxyResetAdaptSchemaBehaviour

	once  *sync.Once
	types map[string]string
}

func (p *MCPToolProxyForceResetInputParamsProcessor) prepareTypesOnce() {
	p.types = make(map[string]string)
	for name, value := range p.Params {
		switch value.(type) {
		case string:
			p.types[name] = "string"
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			p.types[name] = "integer"
		case float32, float64:
			p.types[name] = "number"
		case bool:
			p.types[name] = "boolean"
		case nil:
			p.types[name] = "null"
		case []any:
			p.types[name] = "array"
		case map[string]any:
			p.types[name] = "object"
		default:
			if value == nil {
				p.types[name] = "null"
			} else {
				v := reflect.ValueOf(value)
				switch v.Kind() {
				case reflect.Slice, reflect.Array:
					p.types[name] = "array"
				case reflect.Map, reflect.Struct:
					p.types[name] = "object"
				case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
					reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
					p.types[name] = "integer"
				case reflect.Float32, reflect.Float64:
					p.types[name] = "number"
				case reflect.Bool:
					p.types[name] = "boolean"
				case reflect.String:
					p.types[name] = "string"
				default:
					p.types[name] = "unknown"
				}
			}
		}
	}
}

func (p *MCPToolProxyForceResetInputParamsProcessor) prepareTypes() {
	if p.once == nil {
		p.once = new(sync.Once)
	}
	p.once.Do(p.prepareTypesOnce)
}

func isTypeMatched(actualType string, actualTypes []string, expectedType string) bool {
	norm := func(t string) string {
		if t == "bool" {
			return "boolean"
		}
		return t
	}
	exp := norm(expectedType)
	act := norm(actualType)
	if exp == act && exp != "" && exp != "unknown" {
		return true
	}
	if (exp == "number" && act == "integer") || (exp == "integer" && act == "number") {
		return true
	}
	for _, t := range actualTypes {
		if isTypeMatched(t, nil, expectedType) {
			return true
		}
	}
	return false
}

func (p *MCPToolProxyForceResetInputParamsProcessor) ProcessSchema(tool *mcp.Tool) error {
	p.prepareTypes()

	if tool == nil {
		return nil
	}
	schema, err := proxyInputSchema(tool)
	if err != nil {
		return err
	}
	if schema.Properties == nil {
		schema.Properties = make(map[string]*jsonschema.Schema)
	}

	keys := make([]string, 0, len(p.Params))
	for k := range p.Params {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	switch p.ResetAdaptSchemaBehaviour {
	case MCPToolProxyResetAdaptSchemaBehaviourHide:
		for _, paramName := range keys {
			expectedType := p.types[paramName]
			prop, exists := schema.Properties[paramName]

			if !exists {
				switch p.ResetNotFoundBehaviour {
				case MCPToolProxyResetNotFoundBehaviourError:
					return fmt.Errorf("param is not found: %s", paramName) // TODO: custom error
				case MCPToolProxyResetNotFoundBehaviourSkip, MCPToolProxyResetNotFoundBehaviourSet:
					continue
				}
			} else if !isTypeMatched(prop.Type, prop.Types, expectedType) {
				switch p.ResetNotMatchedTypesBehaviour {
				case MCPToolProxyResetNotMatchedTypesBehaviourError:
					return fmt.Errorf("param %s type mismatch: expected %s, got %s", paramName, expectedType, prop.Type) // TODO: custom error
				case MCPToolProxyResetNotMatchedTypesBehaviourSkip, MCPToolProxyResetNotMatchedTypesBehaviourReset:
					// Proceed with removal
				}
			}

			delete(schema.Properties, paramName)
			schema.Required = slices.DeleteFunc(schema.Required, func(s string) bool {
				return s == paramName
			})
		}
		return nil

	case MCPToolProxyResetAdaptSchemaBehaviourKeep:
		for _, paramName := range keys {
			expectedType := p.types[paramName]
			prop, exists := schema.Properties[paramName]

			if !exists {
				switch p.ResetNotFoundBehaviour {
				case MCPToolProxyResetNotFoundBehaviourError:
					return fmt.Errorf("param is not found: %s", paramName) // TODO: custom error
				case MCPToolProxyResetNotFoundBehaviourSkip, MCPToolProxyResetNotFoundBehaviourSet:
					continue
				}
			} else if !isTypeMatched(prop.Type, prop.Types, expectedType) {
				switch p.ResetNotMatchedTypesBehaviour {
				case MCPToolProxyResetNotMatchedTypesBehaviourError:
					return fmt.Errorf("param %s type mismatch: expected %s, got %s", paramName, expectedType, prop.Type) // TODO: custom error
				case MCPToolProxyResetNotMatchedTypesBehaviourReset:
					prop.Type = expectedType
					prop.Types = nil
				case MCPToolProxyResetNotMatchedTypesBehaviourSkip:
					continue
				}
			}
		}
		return nil

	case MCPToolProxyResetAdaptSchemaBehaviourAnnotate:
		for _, paramName := range keys {
			value := p.Params[paramName]
			expectedType := p.types[paramName]
			prop, exists := schema.Properties[paramName]

			if !exists {
				switch p.ResetNotFoundBehaviour {
				case MCPToolProxyResetNotFoundBehaviourError:
					return fmt.Errorf("param is not found: %s", paramName) // TODO: custom error
				case MCPToolProxyResetNotFoundBehaviourSkip:
					continue
				case MCPToolProxyResetNotFoundBehaviourSet:
					newProp := &jsonschema.Schema{
						Type: expectedType,
					}
					if valBytes, err := json.Marshal(value); err == nil {
						newProp.Default = valBytes
					}
					schema.Properties[paramName] = newProp
					prop = newProp
				}
			} else if !isTypeMatched(prop.Type, prop.Types, expectedType) {
				switch p.ResetNotMatchedTypesBehaviour {
				case MCPToolProxyResetNotMatchedTypesBehaviourError:
					return fmt.Errorf("param %s type mismatch: expected %s, got %s", paramName, expectedType, prop.Type) // TODO: custom error
				case MCPToolProxyResetNotMatchedTypesBehaviourSkip:
					continue
				case MCPToolProxyResetNotMatchedTypesBehaviourReset:
					prop.Type = expectedType
					prop.Types = nil
					if valBytes, err := json.Marshal(value); err == nil {
						prop.Default = valBytes
					}
				}
			} else {
				if valBytes, err := json.Marshal(value); err == nil {
					prop.Default = valBytes
				}
			}

			// Remove from required since it is already preset
			schema.Required = slices.DeleteFunc(schema.Required, func(s string) bool {
				return s == paramName
			})

			// Annotate description to indicate preset value
			if prop != nil {
				presetNote := fmt.Sprintf("[preset: %v]", value)
				if prop.Description == "" {
					prop.Description = presetNote
				} else if !strings.Contains(prop.Description, presetNote) {
					prop.Description = fmt.Sprintf("%s %s", prop.Description, presetNote)
				}
			}
		}
		return nil

	default:
		return fmt.Errorf("unknown reset adapt schema behaviour: %d", p.ResetAdaptSchemaBehaviour)
	}
}

func (p *MCPToolProxyForceResetInputParamsProcessor) ProcessArguments(arguments map[string]any) error {
	maps.Copy(arguments, p.Params)
	return nil
}
