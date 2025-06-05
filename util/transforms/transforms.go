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
	"regexp"
	"strings"
)

var nonAlphanumericRegexp = regexp.MustCompile(`[^a-zA-Z0-9]`)

func TransformStringFunctionStyle(name string) string {
	// Replace spaces with underscores
	name = strings.ReplaceAll(name, " ", "_")

	// Replace non-alphanumeric characters with underscores
	name = nonAlphanumericRegexp.ReplaceAllString(name, "_")

	return strings.ToLower(name)
}
