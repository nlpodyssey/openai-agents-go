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
	"os"
	"strings"
)

func debugFlagEnabled(flag string) bool {
	v, ok := os.LookupEnv(flag)
	return ok && (v == "1" || strings.ToLower(v) == "true")
}

// DontLogModelData - By default we don't log LLM inputs/outputs, to prevent
// exposing sensitive information. Set this flag to enable logging them.
var DontLogModelData = debugFlagEnabled("OPENAI_AGENTS_DONT_LOG_MODEL_DATA")
