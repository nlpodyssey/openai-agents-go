package agents

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/stretchr/testify/require"
)

func TestOpenAITTSModelSerializesDefaultAndExplicitVoice(t *testing.T) {
	for _, test := range []struct {
		name     string
		settings TTSModelSettings
		want     string
	}{
		{name: "default", want: "ash"},
		{name: "explicit", settings: TTSModelSettings{Voice: TTSVoiceCoral}, want: "coral"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var request struct {
				Voice string `json:"voice"`
			}
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, incoming *http.Request) {
				require.NoError(t, json.NewDecoder(incoming.Body).Decode(&request))
				writer.Header().Set("Content-Type", "audio/pcm")
				_, _ = writer.Write([]byte("audio"))
			}))
			t.Cleanup(server.Close)

			model := NewOpenAITTSModel("tts-model", NewOpenaiClient(
				param.NewOpt(server.URL), param.NewOpt("test-key"),
			))
			result := model.Run(t.Context(), "hello", test.settings)
			for range result.Seq() {
			}
			require.NoError(t, result.Error())
			require.Equal(t, test.want, request.Voice)
		})
	}
}
