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

package handoff_prompt

import "fmt"

// RecommendedPromptPrefix is a recommended prompt prefix for agents that use handoffs.
// We recommend including this or similar instructions in any agents that use handoffs.
const RecommendedPromptPrefix = "# System context\n" +
	"You are part of a multi-agent system called the Agents SDK, designed to make agent " +
	"coordination and execution easy. Agents uses two primary abstraction: **Agents** and " +
	"**Handoffs**. An agent encompasses instructions and tools and can hand off a " +
	"conversation to another agent when appropriate. " +
	"Handoffs are achieved by calling a handoff function, generally named " +
	"`transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background;" +
	" do not mention or draw attention to these transfers in your conversation with the user.\n"

// PromptWithHandoffInstructions adds recommended instructions to the prompt for agents that use handoffs.
func PromptWithHandoffInstructions(prompt string) string {
	return fmt.Sprintf("%s\n\n%s", RecommendedPromptPrefix, prompt)
}
