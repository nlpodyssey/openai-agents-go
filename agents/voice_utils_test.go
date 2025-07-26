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
