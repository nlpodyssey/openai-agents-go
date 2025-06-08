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

package runcontext

import (
	"github.com/nlpodyssey/openai-agents-go/usage"
)

// Wrapper wraps the context object that you passed to `Runner().Run()`.
// It also contains information about the usage of the agent run so far.
//
// NOTE: Contexts are not passed to the LLM. They're a way to pass dependencies
// and data to code you implement, like tool functions, callbacks, hooks, etc.
type Wrapper struct {
	// Optional context object, passed by you to `Runner().Run()`.
	Context any

	// The usage of the agent run so far. For streamed responses, the usage
	// will be stale until the last chunk of the stream is processed.
	Usage *usage.Usage
}

func NewWrapper(ctx any) *Wrapper {
	return &Wrapper{
		Context: ctx,
		Usage:   usage.NewUsage(),
	}
}
