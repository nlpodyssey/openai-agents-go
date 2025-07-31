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
	"regexp"
	"strings"
)

// GetTTSSentenceBasedSplitter returns a function that splits text into chunks
// based on sentence boundaries.
//
// minSentenceLength is the minimum length of a sentence to be included in a chunk.
func GetTTSSentenceBasedSplitter(minSentenceLength int) TTSTextSplitterFunc {
	re := regexp.MustCompile(`[.!?]\s+`)
	return func(textBuffer string) (textToProcess, remainingText string, err error) {
		var sentences []string
		start := 0
		for _, match := range re.FindAllStringIndex(textBuffer, -1) {
			end := match[1]
			sentences = append(sentences, strings.TrimSpace(textBuffer[start:end]))
			start = end
		}
		if start < len(textBuffer) {
			sentences = append(sentences, strings.TrimSpace(textBuffer[start:]))
		}

		if len(sentences) > 0 {
			combinedSentences := strings.Join(sentences[:len(sentences)-1], " ")
			if len(combinedSentences) >= minSentenceLength {
				remainingTextBuffer := sentences[len(sentences)-1]
				return combinedSentences, remainingTextBuffer, nil
			}
		}
		return "", textBuffer, nil
	}
}
