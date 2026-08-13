# Provider support

This catalog follows `@earendil-works/pi-ai` as shipped with
`@earendil-works/pi-agent-core` 0.82.1 (Pi commit
`027a5847901b5dde30270abaa1041046cd2b4b55e`). It covers all 10 wire APIs and
all 38 provider IDs in that pinned catalog.

The catalog describes transports and endpoints, not a frozen list of model
names. Pass the model ID accepted by the selected service.

## Quick start

```go
import (
    "os"

    "github.com/z3r2ne/agentcore/provider"
)

model, err := provider.New("openrouter", provider.Config{
    Model:  "anthropic/claude-sonnet-4",
    APIKey: os.Getenv("OPENROUTER_API_KEY"),
})
```

`provider.Lookup` returns one definition and `provider.Builtins` returns a
detached, ID-sorted copy of the whole catalog. Set `Config.API` only when a
provider advertises more than one protocol and the non-preferred protocol is
required.

## Wire APIs

| Wire API | Native package | Notes |
|---|---|---|
| `openai-completions` | `provider/openai` | Chat Completions SSE, parallel tools, reasoning aliases, usage and provider-data replay |
| `openai-responses` | `provider/openairesponses` | Responses SSE, reasoning items, function calls/outputs and cached-token usage |
| `azure-openai-responses` | `provider/openairesponses` | Responses transport with Azure `api-key`; caller supplies the deployment endpoint |
| `openai-codex-responses` | `provider/openairesponses` | Responses transport against the Codex endpoint; caller supplies a valid account token |
| `anthropic-messages` | `provider/anthropic` | Messages SSE, thinking/signatures, tools, images and prompt-cache usage |
| `bedrock-converse-stream` | `provider/bedrock` | AWS SDK ConverseStream, standard AWS credential chain, tools and reasoning signatures |
| `google-generative-ai` | `provider/google` | Gemini SSE, thought signatures, functions and multimodal input |
| `google-vertex` | `provider/google` | Gemini wire format against a caller-supplied Vertex publisher endpoint and authorization header |
| `mistral-conversations` | `provider/mistral` | Mistral conversation compatibility plus deterministic nine-character tool-call IDs |
| `pi-messages` | `provider/pimessages` | Pi/Radius message-event stream |

Every protocol above is implemented. `provider.New` can therefore construct
every catalog entry once its service-specific endpoint and credentials are
available.

## Provider catalog

`Preferred API` is selected by `provider.New` when `Config.API` is empty.
Alternatives are accepted only where listed.

| Provider ID | Preferred API | Other advertised APIs | Endpoint behavior |
|---|---|---|---|
| `amazon-bedrock` | `bedrock-converse-stream` | — | AWS SDK endpoint resolution; optional `BaseURL` override |
| `ant-ling` | `openai-completions` | — | Built in |
| `anthropic` | `anthropic-messages` | — | Built in |
| `azure-openai-responses` | `azure-openai-responses` | — | `BaseURL` required |
| `cerebras` | `openai-completions` | — | Built in |
| `cloudflare-ai-gateway` | `openai-responses` | `anthropic-messages`, `openai-completions` | Account/gateway `BaseURL` required |
| `cloudflare-workers-ai` | `openai-completions` | — | Account-scoped `BaseURL` required |
| `deepseek` | `openai-completions` | — | Built in |
| `fireworks` | `openai-completions` | `anthropic-messages` | Built in |
| `github-copilot` | `openai-responses` | `anthropic-messages`, `openai-completions` | Built in; caller supplies a current Copilot token |
| `google` | `google-generative-ai` | — | Built in |
| `google-vertex` | `google-vertex` | — | Vertex publisher `BaseURL` required |
| `groq` | `openai-completions` | — | Built in |
| `huggingface` | `openai-completions` | — | Built in |
| `kimi-coding` | `anthropic-messages` | — | Built in |
| `minimax` | `anthropic-messages` | — | Built in |
| `minimax-cn` | `anthropic-messages` | — | Built in |
| `mistral` | `mistral-conversations` | — | Built in |
| `moonshotai` | `openai-completions` | — | Built in |
| `moonshotai-cn` | `openai-completions` | — | Built in |
| `nvidia` | `openai-completions` | — | Built in |
| `openai` | `openai-responses` | `openai-completions` | Built in |
| `openai-codex` | `openai-codex-responses` | — | Built in; caller supplies a current Codex token |
| `opencode` | `openai-responses` | `anthropic-messages`, `google-generative-ai`, `openai-completions` | Deployment `BaseURL` required |
| `opencode-go` | `openai-responses` | `anthropic-messages`, `openai-completions` | Deployment `BaseURL` required |
| `openrouter` | `openai-completions` | `openai-responses` | Built in |
| `qwen-token-plan` | `openai-completions` | — | Built in |
| `qwen-token-plan-cn` | `openai-completions` | — | Built in |
| `radius` | `pi-messages` | — | Built in |
| `together` | `openai-completions` | — | Built in |
| `vercel-ai-gateway` | `anthropic-messages` | — | Built in |
| `xai` | `openai-completions` | `openai-responses` | Built in |
| `xiaomi` | `openai-completions` | — | Built in |
| `xiaomi-token-plan-cn` | `openai-completions` | — | Built in |
| `xiaomi-token-plan-ams` | `openai-completions` | — | Built in |
| `xiaomi-token-plan-sgp` | `openai-completions` | — | Built in |
| `zai` | `openai-completions` | — | Built in |
| `zai-coding-cn` | `openai-completions` | — | Built in |

## Credentials and custom endpoints

Credentials are deliberately supplied by the host. The library does not log
in to providers, refresh OAuth tokens, or store secrets.

- HTTP adapters accept `APIKey`, caller-owned `Headers`, and an optional
  `HTTPClient`. A custom `Authorization` header takes precedence where the
  adapter supports bearer authentication.
- Bedrock uses the AWS SDK credential chain. `Region` and `Profile` are
  available in `provider.Config`; `BaseURL` is only for an explicit endpoint
  override.
- Azure requires the complete Responses deployment endpoint as `BaseURL`.
- Vertex requires the publisher base, for example
  `https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/LOCATION/publishers/google`,
  plus a current `Authorization: Bearer ...` header (or the credential form
  accepted by that endpoint).
- Cloudflare and OpenCode entries have account- or deployment-specific URLs,
  so their `BaseURL` cannot be inferred from a provider ID.

Provider endpoints and authentication schemes can change independently of
this module. Production applications should own credential refresh and may
override catalog endpoints without modifying agentcore.
