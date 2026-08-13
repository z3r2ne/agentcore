package provider

import "testing"

func TestPinnedPiProviderCatalog(t *testing.T) {
	builtins := Builtins()
	if len(builtins) != 38 {
		t.Fatalf("providers=%d", len(builtins))
	}
	seen := map[string]bool{}
	apis := map[API]bool{}
	for _, definition := range builtins {
		if seen[definition.ID] || len(definition.APIs) == 0 || definition.PreferredAPI == "" {
			t.Fatalf("invalid definition=%+v", definition)
		}
		seen[definition.ID] = true
		for _, api := range definition.APIs {
			apis[api] = true
		}
	}
	for _, id := range []string{"openai", "anthropic", "google", "google-vertex", "amazon-bedrock", "mistral", "radius", "openrouter"} {
		if !seen[id] {
			t.Errorf("missing %s", id)
		}
	}
	wantAPIs := []API{
		OpenAICompletions, OpenAIResponses, AzureOpenAIResponses, OpenAICodexResponses,
		AnthropicMessages, GoogleGenerativeAI, GoogleVertex, MistralConversations,
		BedrockConverseStream, PiMessages,
	}
	if len(apis) != len(wantAPIs) {
		t.Fatalf("wire APIs=%v", apis)
	}
	for _, api := range wantAPIs {
		if !apis[api] {
			t.Errorf("missing wire API %s", api)
		}
	}
}
