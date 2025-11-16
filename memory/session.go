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

package memory

import (
	"context"

	"github.com/openai/openai-go/v3/responses"
)

// TResponseInputItem is a type alias for the ResponseInputItemUnionParam type from the OpenAI SDK.
type TResponseInputItem = responses.ResponseInputItemUnionParam

// A Session stores conversation history for a specific session, allowing
// agents to maintain context without requiring explicit manual memory management.
type Session interface {
	SessionID(context.Context) string

	// GetItems retrieves the conversation history for this session.
	//
	// `limit` is the maximum number of items to retrieve. If <= 0, retrieves all items.
	// When specified, returns the latest N items in chronological order.
	GetItems(ctx context.Context, limit int) ([]TResponseInputItem, error)

	// AddItems adds new items to the conversation history.
	AddItems(ctx context.Context, items []TResponseInputItem) error

	// PopItem removes and returns the most recent item from the session.
	// It returns nil if the session is empty.
	PopItem(context.Context) (*TResponseInputItem, error)

	// ClearSession clears all items for this session.
	ClearSession(context.Context) error
}
