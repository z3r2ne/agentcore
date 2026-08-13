// Package provider contains the built-in provider catalog and constructors for
// agentcore's native wire-protocol adapters.
package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/z3r2ne/agentcore"
	anthropicprovider "github.com/z3r2ne/agentcore/provider/anthropic"
	bedrockprovider "github.com/z3r2ne/agentcore/provider/bedrock"
	googleprovider "github.com/z3r2ne/agentcore/provider/google"
	mistralprovider "github.com/z3r2ne/agentcore/provider/mistral"
	chatprovider "github.com/z3r2ne/agentcore/provider/openai"
	responsesprovider "github.com/z3r2ne/agentcore/provider/openairesponses"
	pimessagesprovider "github.com/z3r2ne/agentcore/provider/pimessages"
)

// API identifies a provider wire protocol.
type API string

const (
	OpenAICompletions     API = "openai-completions"
	OpenAIResponses       API = "openai-responses"
	AzureOpenAIResponses  API = "azure-openai-responses"
	OpenAICodexResponses  API = "openai-codex-responses"
	AnthropicMessages     API = "anthropic-messages"
	GoogleGenerativeAI    API = "google-generative-ai"
	GoogleVertex          API = "google-vertex"
	MistralConversations  API = "mistral-conversations"
	BedrockConverseStream API = "bedrock-converse-stream"
	PiMessages            API = "pi-messages"
)

// Definition is lightweight catalog metadata. APIs lists every protocol used
// by the provider's Pi catalog; PreferredAPI is used by New unless overridden.
type Definition struct {
	ID              string   `json:"id"`
	Name            string   `json:"name"`
	BaseURL         string   `json:"baseUrl,omitempty"`
	PreferredAPI    API      `json:"preferredApi"`
	APIs            []API    `json:"apis"`
	EnvironmentKeys []string `json:"environmentKeys,omitempty"`
}

// Config supplies credentials and transport for a catalog provider.
type Config struct {
	Model      string
	API        API
	APIKey     string
	BaseURL    string
	Headers    http.Header
	HTTPClient *http.Client
	// Region and Profile are used by Amazon Bedrock's standard AWS credential chain.
	Region  string
	Profile string
}

// ErrUnsupportedAPI means a catalog entry references a future wire protocol
// that this version of agentcore does not implement.
var ErrUnsupportedAPI = errors.New("provider: wire API is not implemented")

var definitions = []Definition{
	{ID: "amazon-bedrock", Name: "Amazon Bedrock", PreferredAPI: BedrockConverseStream, APIs: []API{BedrockConverseStream}, EnvironmentKeys: []string{"AWS_BEARER_TOKEN_BEDROCK", "AWS_PROFILE", "AWS_ACCESS_KEY_ID"}},
	{ID: "ant-ling", Name: "Ant Ling", BaseURL: "https://api.ant-ling.com/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"ANT_LING_API_KEY"}},
	{ID: "anthropic", Name: "Anthropic", BaseURL: "https://api.anthropic.com", PreferredAPI: AnthropicMessages, APIs: []API{AnthropicMessages}, EnvironmentKeys: []string{"ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"}},
	{ID: "azure-openai-responses", Name: "Azure OpenAI", PreferredAPI: AzureOpenAIResponses, APIs: []API{AzureOpenAIResponses}, EnvironmentKeys: []string{"AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL"}},
	{ID: "cerebras", Name: "Cerebras", BaseURL: "https://api.cerebras.ai/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"CEREBRAS_API_KEY"}},
	{ID: "cloudflare-ai-gateway", Name: "Cloudflare AI Gateway", PreferredAPI: OpenAIResponses, APIs: []API{AnthropicMessages, OpenAICompletions, OpenAIResponses}, EnvironmentKeys: []string{"CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_GATEWAY_ID"}},
	{ID: "cloudflare-workers-ai", Name: "Cloudflare Workers AI", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID"}},
	{ID: "deepseek", Name: "DeepSeek", BaseURL: "https://api.deepseek.com", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"DEEPSEEK_API_KEY"}},
	{ID: "fireworks", Name: "Fireworks", BaseURL: "https://api.fireworks.ai/inference", PreferredAPI: OpenAICompletions, APIs: []API{AnthropicMessages, OpenAICompletions}, EnvironmentKeys: []string{"FIREWORKS_API_KEY"}},
	{ID: "github-copilot", Name: "GitHub Copilot", BaseURL: "https://api.individual.githubcopilot.com", PreferredAPI: OpenAIResponses, APIs: []API{AnthropicMessages, OpenAICompletions, OpenAIResponses}, EnvironmentKeys: []string{"GITHUB_COPILOT_TOKEN"}},
	{ID: "google", Name: "Google", BaseURL: "https://generativelanguage.googleapis.com/v1beta", PreferredAPI: GoogleGenerativeAI, APIs: []API{GoogleGenerativeAI}, EnvironmentKeys: []string{"GEMINI_API_KEY"}},
	{ID: "google-vertex", Name: "Google Vertex AI", PreferredAPI: GoogleVertex, APIs: []API{GoogleVertex}, EnvironmentKeys: []string{"GOOGLE_CLOUD_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"}},
	{ID: "groq", Name: "Groq", BaseURL: "https://api.groq.com/openai/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"GROQ_API_KEY"}},
	{ID: "huggingface", Name: "Hugging Face", BaseURL: "https://router.huggingface.co/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"HF_TOKEN"}},
	{ID: "kimi-coding", Name: "Kimi For Coding", BaseURL: "https://api.kimi.com/coding", PreferredAPI: AnthropicMessages, APIs: []API{AnthropicMessages}, EnvironmentKeys: []string{"KIMI_API_KEY"}},
	{ID: "minimax", Name: "MiniMax", BaseURL: "https://api.minimax.io/anthropic", PreferredAPI: AnthropicMessages, APIs: []API{AnthropicMessages}, EnvironmentKeys: []string{"MINIMAX_API_KEY"}},
	{ID: "minimax-cn", Name: "MiniMax CN", BaseURL: "https://api.minimaxi.com/anthropic", PreferredAPI: AnthropicMessages, APIs: []API{AnthropicMessages}, EnvironmentKeys: []string{"MINIMAX_CN_API_KEY"}},
	{ID: "mistral", Name: "Mistral", BaseURL: "https://api.mistral.ai/v1", PreferredAPI: MistralConversations, APIs: []API{MistralConversations}, EnvironmentKeys: []string{"MISTRAL_API_KEY"}},
	{ID: "moonshotai", Name: "Moonshot AI", BaseURL: "https://api.moonshot.ai/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"MOONSHOT_API_KEY"}},
	{ID: "moonshotai-cn", Name: "Moonshot AI CN", BaseURL: "https://api.moonshot.cn/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"MOONSHOT_API_KEY"}},
	{ID: "nvidia", Name: "NVIDIA", BaseURL: "https://integrate.api.nvidia.com/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"NVIDIA_API_KEY"}},
	{ID: "openai", Name: "OpenAI", BaseURL: "https://api.openai.com/v1", PreferredAPI: OpenAIResponses, APIs: []API{OpenAIResponses, OpenAICompletions}, EnvironmentKeys: []string{"OPENAI_API_KEY"}},
	{ID: "openai-codex", Name: "OpenAI Codex", BaseURL: "https://chatgpt.com/backend-api", PreferredAPI: OpenAICodexResponses, APIs: []API{OpenAICodexResponses}, EnvironmentKeys: []string{"OPENAI_CODEX_TOKEN"}},
	{ID: "opencode", Name: "OpenCode Zen", PreferredAPI: OpenAIResponses, APIs: []API{AnthropicMessages, GoogleGenerativeAI, OpenAICompletions, OpenAIResponses}, EnvironmentKeys: []string{"OPENCODE_API_KEY"}},
	{ID: "opencode-go", Name: "OpenCode Go", PreferredAPI: OpenAIResponses, APIs: []API{AnthropicMessages, OpenAICompletions, OpenAIResponses}, EnvironmentKeys: []string{"OPENCODE_API_KEY"}},
	{ID: "openrouter", Name: "OpenRouter", BaseURL: "https://openrouter.ai/api/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions, OpenAIResponses}, EnvironmentKeys: []string{"OPENROUTER_API_KEY"}},
	{ID: "qwen-token-plan", Name: "Qwen Token Plan", BaseURL: "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"QWEN_TOKEN_PLAN_API_KEY"}},
	{ID: "qwen-token-plan-cn", Name: "Qwen Token Plan CN", BaseURL: "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"QWEN_TOKEN_PLAN_API_KEY"}},
	{ID: "radius", Name: "Radius", BaseURL: "https://radius.pi.dev", PreferredAPI: PiMessages, APIs: []API{PiMessages}, EnvironmentKeys: []string{"RADIUS_API_KEY"}},
	{ID: "together", Name: "Together AI", BaseURL: "https://api.together.ai/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"TOGETHER_API_KEY"}},
	{ID: "vercel-ai-gateway", Name: "Vercel AI Gateway", BaseURL: "https://ai-gateway.vercel.sh", PreferredAPI: AnthropicMessages, APIs: []API{AnthropicMessages}, EnvironmentKeys: []string{"AI_GATEWAY_API_KEY"}},
	{ID: "xai", Name: "xAI", BaseURL: "https://api.x.ai/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions, OpenAIResponses}, EnvironmentKeys: []string{"XAI_API_KEY"}},
	{ID: "xiaomi", Name: "Xiaomi", BaseURL: "https://api.xiaomimimo.com/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"XIAOMI_API_KEY"}},
	{ID: "xiaomi-token-plan-cn", Name: "Xiaomi Token Plan CN", BaseURL: "https://token-plan-cn.xiaomimimo.com/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"XIAOMI_API_KEY"}},
	{ID: "xiaomi-token-plan-ams", Name: "Xiaomi Token Plan AMS", BaseURL: "https://token-plan-ams.xiaomimimo.com/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"XIAOMI_API_KEY"}},
	{ID: "xiaomi-token-plan-sgp", Name: "Xiaomi Token Plan SGP", BaseURL: "https://token-plan-sgp.xiaomimimo.com/v1", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"XIAOMI_API_KEY"}},
	{ID: "zai", Name: "Z.AI", BaseURL: "https://api.z.ai/api/coding/paas/v4", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"ZAI_API_KEY"}},
	{ID: "zai-coding-cn", Name: "Z.AI Coding CN", BaseURL: "https://open.bigmodel.cn/api/coding/paas/v4", PreferredAPI: OpenAICompletions, APIs: []API{OpenAICompletions}, EnvironmentKeys: []string{"ZAI_API_KEY"}},
}

// Builtins returns detached provider definitions ordered by ID.
func Builtins() []Definition {
	result := make([]Definition, len(definitions))
	for i, d := range definitions {
		d.APIs = append([]API(nil), d.APIs...)
		d.EnvironmentKeys = append([]string(nil), d.EnvironmentKeys...)
		result[i] = d
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result
}

// Lookup returns detached catalog metadata.
func Lookup(id string) (Definition, bool) {
	id = strings.TrimSpace(id)
	for _, d := range definitions {
		if d.ID == id {
			d.APIs = append([]API(nil), d.APIs...)
			d.EnvironmentKeys = append([]string(nil), d.EnvironmentKeys...)
			return d, true
		}
	}
	return Definition{}, false
}

// New constructs a native model for a catalog provider. BaseURL is required
// for account-scoped gateways whose endpoint cannot be known statically.
func New(id string, config Config) (agentcore.Model, error) {
	definition, ok := Lookup(id)
	if !ok {
		return nil, fmt.Errorf("provider: unknown provider %q", id)
	}
	api := config.API
	if api == "" {
		api = definition.PreferredAPI
	}
	supported := false
	for _, candidate := range definition.APIs {
		if candidate == api {
			supported = true
			break
		}
	}
	if !supported {
		return nil, fmt.Errorf("provider: %s does not advertise API %s", id, api)
	}
	base := strings.TrimSpace(config.BaseURL)
	if base == "" {
		base = definition.BaseURL
	}
	if base == "" && api != BedrockConverseStream {
		return nil, fmt.Errorf("provider: BaseURL is required for %s", id)
	}
	switch api {
	case OpenAICompletions:
		return chatprovider.New(chatprovider.Config{Model: config.Model, APIKey: config.APIKey, BaseURL: base, Header: config.Headers, HTTPClient: config.HTTPClient})
	case MistralConversations:
		return mistralprovider.New(mistralprovider.Config{Model: config.Model, APIKey: config.APIKey, BaseURL: base, Headers: config.Headers, HTTPClient: config.HTTPClient})
	case OpenAIResponses, AzureOpenAIResponses, OpenAICodexResponses:
		headers := config.Headers.Clone()
		apiKey := config.APIKey
		if api == OpenAICodexResponses && !strings.HasSuffix(strings.TrimRight(base, "/"), "/codex") && !strings.HasSuffix(strings.TrimRight(base, "/"), "/codex/responses") {
			base = strings.TrimRight(base, "/") + "/codex"
		}
		if api == AzureOpenAIResponses && apiKey != "" {
			if headers == nil {
				headers = make(http.Header)
			}
			headers.Set("api-key", apiKey)
			apiKey = ""
		}
		return responsesprovider.New(responsesprovider.Config{Model: config.Model, APIKey: apiKey, BaseURL: base, Headers: headers, HTTPClient: config.HTTPClient})
	case AnthropicMessages:
		return anthropicprovider.New(anthropicprovider.Config{Model: config.Model, APIKey: config.APIKey, BaseURL: base, Headers: config.Headers, HTTPClient: config.HTTPClient})
	case GoogleGenerativeAI, GoogleVertex:
		return googleprovider.New(googleprovider.Config{Model: config.Model, APIKey: config.APIKey, BaseURL: base, Headers: config.Headers, HTTPClient: config.HTTPClient})
	case BedrockConverseStream:
		return bedrockprovider.New(context.Background(), bedrockprovider.Config{Model: config.Model, Region: config.Region, Profile: config.Profile, BaseEndpoint: base})
	case PiMessages:
		return pimessagesprovider.New(pimessagesprovider.Config{Model: config.Model, APIKey: config.APIKey, BaseURL: base, Headers: config.Headers, HTTPClient: config.HTTPClient})
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedAPI, api)
	}
}
