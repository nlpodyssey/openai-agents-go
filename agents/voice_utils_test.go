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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetTTSSentenceBasedSplitter(t *testing.T) {
	splitterFunc := GetTTSSentenceBasedSplitter(10)

	testCases := []struct {
		textBuffer        string
		wantTextToProcess string
		wantRemainingText string
	}{
		{"", "", ""},
		{"foo", "", "foo"},
		{"foo.", "", "foo."},
		{"foo. ", "", "foo. "},
		{"foo. bar! baz? quux.", "foo. bar! baz?", "quux."},
		{"foo. \n bar! \t baz?   quux.", "foo. bar! baz?", "quux."},
		{"foo1 \n bar2 \t baz3   quux4", "", "foo1 \n bar2 \t baz3   quux4"},
	}

	for _, tt := range testCases {
		t.Run(fmt.Sprintf("%q", tt.textBuffer), func(t *testing.T) {
			gotTextToProcess, gotRemainingText, err := splitterFunc(tt.textBuffer)
			require.NoError(t, err)
			assert.Equal(t, tt.wantTextToProcess, gotTextToProcess)
			assert.Equal(t, tt.wantRemainingText, gotRemainingText)
		})
	}
}
