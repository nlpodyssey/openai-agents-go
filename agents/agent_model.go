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

type AgentModel struct {
	s *string
	m *Model
}

func NewAgentModelName(modelName string) AgentModel {
	return AgentModel{s: &modelName}
}

func NewAgentModel(m Model) AgentModel {
	if m == nil {
		panic("Model cannot be nil")
	}
	return AgentModel{m: &m}
}

func (am AgentModel) IsModelName() bool {
	return am.s != nil
}

func (am AgentModel) IsModel() bool {
	return am.m != nil
}

func (am AgentModel) SafeModelName() (string, bool) {
	if am.IsModelName() {
		return *am.s, true
	}
	return "", false
}

func (am AgentModel) SafeModel() (Model, bool) {
	if am.IsModel() {
		return *am.m, true
	}
	return nil, false
}

func (am AgentModel) ModelName() string {
	if !am.IsModelName() {
		panic("AgentModel is not of type ModelName")
	}
	return *am.s
}

func (am AgentModel) Model() Model {
	if !am.IsModel() {
		panic("AgentModel is not of type Model")
	}
	return *am.m
}
