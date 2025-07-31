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

//TODO # If we create a new httpx client for each request, that would mean no sharing of connection pools,
//# which would mean worse latency and resource usage. So, we share the client across requests.
//def shared_http_client() -> httpx.AsyncClient:
//    global _http_client
//    if _http_client is None:
//        _http_client = DefaultAsyncHttpxClient()
//    return _http_client
//
//
//DEFAULT_STT_MODEL = "gpt-4o-transcribe"
//DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"

// OpenAIVoiceModelProvider is a voice model provider that uses OpenAI models.
type OpenAIVoiceModelProvider struct {
}

type OpenAIVoiceModelProviderParams struct {
	//api_key: str | None = None,
	//base_url: str | None = None,
	//openai_client: AsyncOpenAI | None = None,
	//organization: str | None = None,
	//project: str | None = None,
}

func NewDefaultOpenAIVoiceModelProvider() *OpenAIVoiceModelProvider {
	return NewOpenAIVoiceModelProvider(OpenAIVoiceModelProviderParams{})
}

func NewOpenAIVoiceModelProvider(params OpenAIVoiceModelProviderParams) *OpenAIVoiceModelProvider {
	//        """Create a new OpenAI voice model provider.
	//
	//        Args:
	//            api_key: The API key to use for the OpenAI client. If not provided, we will use the
	//                default API key.
	//            base_url: The base URL to use for the OpenAI client. If not provided, we will use the
	//                default base URL.
	//            openai_client: An optional OpenAI client to use. If not provided, we will create a new
	//                OpenAI client using the api_key and base_url.
	//            organization: The organization to use for the OpenAI client.
	//            project: The project to use for the OpenAI client.
	//        """
	//        if openai_client is not None:
	//            assert api_key is None and base_url is None, (
	//                "Don't provide api_key or base_url if you provide openai_client"
	//            )
	//            self._client: AsyncOpenAI | None = openai_client
	//        else:
	//            self._client = None
	//            self._stored_api_key = api_key
	//            self._stored_base_url = base_url
	//            self._stored_organization = organization
	//            self._stored_project = project
	panic("implement me") //TODO implement me
}

//    # We lazy load the client in case you never actually use OpenAIProvider(). Otherwise
//    # AsyncOpenAI() raises an error if you don't have an API key set.
//    def _get_client(self) -> AsyncOpenAI:
//        if self._client is None:
//            self._client = _openai_shared.get_default_openai_client() or AsyncOpenAI(
//                api_key=self._stored_api_key or _openai_shared.get_default_openai_key(),
//                base_url=self._stored_base_url,
//                organization=self._stored_organization,
//                project=self._stored_project,
//                http_client=shared_http_client(),
//            )
//
//        return self._client
//

func (p *OpenAIVoiceModelProvider) GetSTTModel(modelName string) (STTModel, error) {
	//def get_stt_model(self, model_name: str | None) -> STTModel:
	//    """Get a speech-to-text model by name.
	//
	//    Args:
	//        model_name: The name of the model to get.
	//
	//    Returns:
	//        The speech-to-text model.
	//    """
	//    return OpenAISTTModel(model_name or DEFAULT_STT_MODEL, self._get_client())
	panic("implement me") //TODO implement me
}

func (p *OpenAIVoiceModelProvider) GetTTSModel(modelName string) (TTSModel, error) {
	//def get_tts_model(self, model_name: str | None) -> TTSModel:
	//    """Get a text-to-speech model by name.
	//
	//    Args:
	//        model_name: The name of the model to get.
	//
	//    Returns:
	//        The text-to-speech model.
	//    """
	//    return OpenAITTSModel(model_name or DEFAULT_TTS_MODEL, self._get_client())
	panic("implement me") //TODO implement me
}
