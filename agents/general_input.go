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
	"fmt"
	"slices"
)

// Input can be either a string or a list of TResponseInputItem.
type Input interface {
	isInput()
}

type InputString string

func (InputString) isInput()         {}
func (s InputString) String() string { return string(s) }

type InputItems []TResponseInputItem

func (InputItems) isInput() {}

func (items InputItems) Copy() InputItems {
	return slices.Clone(items)
}

func CopyInput(input Input) Input {
	switch v := input.(type) {
	case InputString:
		return v
	case InputItems:
		return v.Copy()
	default:
		// This would be an unrecoverable implementation bug, so a panic is appropriate.
		panic(fmt.Errorf("unexpected Input type %T", v))
	}
}
