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

package transforms

import (
	"os"
	"regexp"
	"strings"
)

// NamingConvention represents different naming conventions
type NamingConvention string

const (
	SnakeCase NamingConvention = "snake_case"
	CamelCase NamingConvention = "camelCase"
)

var (
	// currentConvention is set once at initialization from OPENAI_AGENTS_NAMING_CONVENTION
	// Default is "snake_case", can be "snake_case" or "camelCase"
	currentConvention = initNamingConvention()

	nonAlphanumericRegexp = regexp.MustCompile(`[^a-zA-Z0-9]`)
)

func initNamingConvention() NamingConvention {
	convention := os.Getenv("OPENAI_AGENTS_NAMING_CONVENTION")
	switch convention {
	case "camelCase":
		return CamelCase
	case "snake_case":
		return SnakeCase
	default:
		return SnakeCase
	}
}

// ToCase converts a string to the specified naming convention
func ToCase(name string) string {
	switch currentConvention {
	case CamelCase:
		return ToCamelCase(name)
	case SnakeCase:
		return ToSnakeCase(name)
	default:
		return ToSnakeCase(name) // default to snake_case
	}
}

// ApplyCase converts a string to a specific naming convention
// (in case you need to override the default)
func ApplyCase(name string, convention NamingConvention) string {
	switch convention {
	case CamelCase:
		return ToCamelCase(name)
	case SnakeCase:
		return ToSnakeCase(name)
	default:
		return ToSnakeCase(name)
	}
}

// GetCurrentConvention returns the current naming convention
func GetCurrentConvention() NamingConvention {
	return currentConvention
}

// ToCamelCase converts various naming formats to camelCase
// Examples:
//   - snake_case -> snakeCase
//   - PascalCase -> pascalCase
//   - alreadyCamelCase -> alreadyCamelCase
func ToCamelCase(name string) string {
	if name == "" {
		return name
	}

	// Handle snake_case conversion
	if strings.Contains(name, "_") {
		parts := strings.Split(name, "_")
		if len(parts) == 0 {
			return name
		}

		result := strings.ToLower(parts[0])
		for i := 1; i < len(parts); i++ {
			if parts[i] != "" {
				result += strings.Title(strings.ToLower(parts[i]))
			}
		}
		return result
	}

	// Handle PascalCase to camelCase conversion
	if len(name) > 0 && name[0] >= 'A' && name[0] <= 'Z' {
		return strings.ToLower(name[:1]) + name[1:]
	}

	return name
}

// ToSnakeCase converts various naming formats to snake_case
// Examples:
//   - camelCase -> camel_case
//   - PascalCase -> pascal_case
//   - already_snake_case -> already_snake_case
func ToSnakeCase(name string) string {
	if name == "" {
		return name
	}

	var result strings.Builder
	for i, r := range name {
		// Handle uppercase letters
		if r >= 'A' && r <= 'Z' {
			// Insert underscore before uppercase letters (except first character)
			if i > 0 {
				// Check if previous character was lowercase to avoid double underscores
				if name[i-1] >= 'a' && name[i-1] <= 'z' {
					result.WriteRune('_')
				}
			}
			result.WriteRune(r)
		} else {
			result.WriteRune(r)
		}
	}

	return strings.ToLower(result.String())
}

func TransformStringFunctionStyle(name string) string {
	// Replace spaces with underscores
	name = strings.ReplaceAll(name, " ", "_")

	// Replace non-alphanumeric characters with underscores
	name = nonAlphanumericRegexp.ReplaceAllString(name, "_")

	return strings.ToLower(name)
}
