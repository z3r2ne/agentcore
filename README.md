# agentcore

`agentcore` is a provider-neutral Go implementation of Pi's core agent-loop
semantics. It is designed to be imported as a library rather than launched as
a CLI or RPC subprocess.

Documentation: [中文使用指南](docs/USAGE.zh-CN.md) · [Pi conformance boundary](PI_CONFORMANCE.md)

## Installation

```bash
go get github.com/z3r2ne/agentcore@v0.2.1
```

For a private repository, bypass the public Go proxy/checksum database and
route GitHub HTTPS module fetches through SSH on port 443:

```bash
go env -w GOPRIVATE=github.com/z3r2ne/*
git config --global url."ssh://git@ssh.github.com:443/".insteadOf https://github.com/
go get github.com/z3r2ne/agentcore@v0.2.1
```

Use a read-only deploy key, GitHub App token, or another least-privilege
credential in CI. For local sibling development, use:

```go
require github.com/z3r2ne/agentcore v0.0.0

replace github.com/z3r2ne/agentcore => ../agentcore
```

It includes:

- streaming assistant, thinking, and tool-call deltas;
- repeated model → tool → model turns;
- parallel or sequential tool execution;
- deterministic tool-result ordering;
- `context.Context` cancellation;
- Pi-style lifecycle events;
- argument validation and truncated-tool-call protection;
- lifecycle hooks and context transformation;
- ordered lifecycle interceptor chains for policy, approval, audit, caching,
  argument rewriting, result filtering, and telemetry;
- dynamic Skills that contribute trusted instructions, tools, and
  interceptors at Agent build time;
- a fluent Builder for execution-scoped composition;
- automatic JSON Schema argument validation;
- model retry events and tool timeout/retry/panic protection;
- context budgeting, tool-result truncation, and pluggable compaction;
- stateful sessions with steering, follow-up, abort, and subscriptions;
- multimodal image, audio, video, and file tool results;
- run and session usage aggregation;
- synchronous callback and asynchronous iterator APIs;
- an Eino model/tool adapter.

It intentionally excludes JSONL persistence, RPC, TUI, authentication,
provider catalogs, extensions, and platform-specific coding tools.

The root package has no dependency on Aegis, Coordination, Phone, MCP, HTTP,
or any application database. Applications own those integrations and expose
them as `Model`, `Tool`, `Skill`, `Interceptor`, or `SessionStore` adapters.

## Composition

For reusable applications, prefer `Builder`: it assembles one immutable,
concurrency-safe Agent from project tools, dynamically loaded Skills, and
interceptors.

```go
audit := agentcore.InterceptorFuncs{
    Name: "audit",
    BeforeTool: func(ctx context.Context, call agentcore.ToolCallContext) (agentcore.ToolCallDecision, error) {
        log.Printf("calling %s with %s", call.Call.Name, call.Call.Arguments)
        return agentcore.ToolCallDecision{}, nil
    },
    AfterTool: func(ctx context.Context, call agentcore.ToolCallContext, result *agentcore.ToolResult) error {
        log.Printf("finished %s executed=%t attempts=%d", call.Call.Name, call.Executed, call.Attempts)
        return nil
    },
}

reviewSkill := agentcore.FuncSkill{
    SkillDefinition: agentcore.SkillDefinition{Name: "review", Version: "1"},
    Content: agentcore.SkillContent{
        Instructions: "Review changes for correctness and cite concrete evidence.",
        Tools:        []agentcore.Tool{diffTool, testTool},
    },
}

agent, err := agentcore.NewBuilder(model).
    SystemPrompt("You are a software engineering Agent.").
    Tools(projectSearchTool).
    Skills(reviewSkill).
    Use(audit).
    Configure(func(config *agentcore.Config) {
        config.MaxTurns = 80
        config.MaxToolConcurrency = 4
    }).
    Build(ctx)
```

`Skill.Load` may read a `SKILL.md`, database record, embedded asset, or remote
registry. Loading happens during `Build`, not inside the model/tool loop. This
keeps the core deterministic while allowing each project to select Skills per
execution.

## Interceptors

An interceptor implements `InterceptorName` plus any optional lifecycle
interfaces it needs:

- `BeforeModelCallInterceptor` / `AfterModelCallInterceptor`;
- `BeforeToolCallInterceptor` / `AfterToolCallInterceptor`;
- `PrepareNextTurnInterceptor`;
- `ShouldStopInterceptor`.

Before interceptors run in registration order; after interceptors run in
reverse order, matching normal middleware nesting. A before-tool interceptor
may block a call or replace its JSON arguments before schema validation. An
after-tool interceptor runs even for blocked, unknown, or schema-invalid calls;
`ToolCallContext.Executed` distinguishes actual execution. It may transform the
result returned to the model.

Interceptors attached to an Agent may run concurrently when the Agent is reused
or tools execute in parallel. Implementations must protect their own mutable
state. Use the callback `context.Context` and
`ToolInvocationFromContext` for execution-local correlation.

## Core API

```go
tool := agentcore.FuncTool{
    ToolDefinition: agentcore.ToolDefinition{
        Name:        "weather",
        Description: "Get the weather for a city",
        Parameters: json.RawMessage(`{
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }`),
    },
    ExecuteFunc: func(
        ctx context.Context,
        arguments json.RawMessage,
        update agentcore.ToolUpdateSink,
    ) (agentcore.ToolResult, error) {
        return agentcore.TextToolResult("Sunny, 24 C"), nil
    },
}

agent, err := agentcore.New(agentcore.Config{
    Model:        modelAdapter,
    SystemPrompt: "You are a concise assistant.",
    Tools:        []agentcore.Tool{tool},
})
if err != nil {
    log.Fatal(err)
}

stream := agent.Stream(
    context.Background(),
    agentcore.State{},
    []agentcore.Message{agentcore.TextMessage(agentcore.RoleUser, "Weather in Shanghai?")},
)

for {
    event, ok := stream.Next()
    if !ok {
        break
    }
    if event.Type == agentcore.EventMessageUpdate && event.Delta != nil && event.Delta.TextDelta != "" {
        fmt.Print(event.Delta.TextDelta)
    }
}

result, err := stream.Result()
```

For callers that already own an event bus, `Agent.Prompt` invokes an
`EventSink` synchronously instead of allocating an iterator.

Tool arguments remain observable while they are generated. On
`EventMessageUpdate`, read `event.Delta.ToolCallDeltas`: each
`ArgumentsDelta` is arbitrary text and callers may concatenate it by tool-call
index for a typewriter preview. `event.Message.ToolCalls()` exposes no partial
`Arguments`; final, valid JSON arguments appear on `EventMessageEnd` and the
subsequent `EventToolExecutionStart`.

## OpenAI-compatible provider

The optional `provider/openai` package implements streaming Chat Completions
without adding provider behavior to the root package:

```go
import (
    "context"
    "os"
    "time"

    "github.com/z3r2ne/agentcore"
    openaiprovider "github.com/z3r2ne/agentcore/provider/openai"
)

model, err := openaiprovider.New(openaiprovider.Config{
    Model:   "gpt-4.1",
    APIKey:  os.Getenv("OPENAI_API_KEY"),
    BaseURL: "https://api.openai.com/v1", // replace for a compatible service
    Headers: map[string]string{"OpenAI-Organization": "org_example"},
})
if err != nil {
    panic(err)
}

agent, err := agentcore.New(agentcore.Config{
    Model: model,
    ModelRetry: agentcore.RetryPolicy{
        MaxAttempts:  3,
        InitialDelay: 250 * time.Millisecond,
        ShouldRetry:  openaiprovider.IsRetryable,
    },
})

result, err := agent.Prompt(context.Background(), agentcore.State{}, []agentcore.Message{
    agentcore.TextMessage(agentcore.RoleUser, "Use tools when useful."),
}, nil)
```

The adapter supports SSE, parallel function calls, `reasoning_content`, usage,
serializable provider data across tool turns, custom endpoints, headers and HTTP
clients, and context cancellation. Non-2xx errors are returned as
`*openai.Error`; body, SSE-event, and total-response limits are configurable.
Request `ModelOptions` are forwarded except transport-owned fields such as
`model`, `messages`, `tools`, `stream`, `stream_options`, and `n`.

## Eino adapter

Construct any Eino `model.ToolCallingChatModel`, then wrap it:

```go
chatModel, err := openai.NewChatModel(ctx, &openai.ChatModelConfig{
    Model:  "gpt-4.1",
    APIKey: os.Getenv("OPENAI_API_KEY"),
})
if err != nil {
    log.Fatal(err)
}

coreModel := einoadapter.Model{ChatModel: chatModel}
agent, err := agentcore.New(agentcore.Config{Model: coreModel})
```

Existing Eino tools can be converted once during startup:

```go
coreTool, err := einoadapter.NewTool(ctx, einoTool)
```

Run-local settings such as thinking level, provider transport, session cache
keys, or refreshed credentials can be carried through `Config.ModelOptions`.
Map them to Eino options at request time:

```go
coreModel := einoadapter.Model{
    ChatModel: chatModel,
    Options: func(ctx context.Context, values map[string]any) ([]model.Option, error) {
        return resolveEinoOptions(ctx, values)
    },
}
```

The Eino adapter preserves the merged raw provider message in memory across
tool turns. Provider-specific reasoning signatures and metadata therefore
survive the next model request without becoming part of the public core API.
The serialized representation also survives JSON round-trips of `State`.

## Stateful session

`Session` owns conversation state and permits messages while a run is active:

```go
session, err := agentcore.NewSession(agent, agentcore.State{}, agentcore.SessionOptions{
    SteeringMode: agentcore.DeliveryOne,
    FollowUpMode: agentcore.DeliveryAll,
})

run := session.Stream(ctx, []agentcore.Message{
    agentcore.TextMessage(agentcore.RoleUser, "Start the task"),
})

session.Steer(agentcore.TextMessage(agentcore.RoleUser, "Use the safer approach"))
session.FollowUp(agentcore.TextMessage(agentcore.RoleUser, "Now summarize"))

result, err := run.Result()
```

`Session.Abort`, `WaitForIdle`, `Status`, `State`, and `Subscribe` provide the
stateful controls normally supplied by a long-running agent process.

## SQLite persistence

SQLite persistence is optional and isolated in `agentcore/sqlitestore`:

```go
store, err := sqlitestore.Open("./data/agent-sessions.db")
if err != nil {
    log.Fatal(err)
}
defer store.Close()

session, err := agentcore.NewSession(agent, agentcore.State{}, agentcore.SessionOptions{
    Store:     store,
    SessionID: "conversation-123",
})
```

Completed runs and accepted steering/follow-up messages are checkpointed
automatically. Restore the same conversation after restart with:

```go
session, err := store.RestoreSession(ctx, "conversation-123", agent, agentcore.SessionOptions{})
```

The table stores one atomically replaced, versioned JSON snapshot per session,
including message history, serialized provider data, usage, delivery modes, and
undelivered queues. `ListSessions`, `LoadSession`, `SaveSession`, and
`DeleteSession` are also available for explicit lifecycle management.

## Execution policies

```go
agent, err := agentcore.New(agentcore.Config{
    Model: modelAdapter,
    ModelRetry: agentcore.RetryPolicy{
        MaxAttempts:  3,
        InitialDelay: 200 * time.Millisecond,
        MaxDelay:     2 * time.Second,
    },
    ToolPolicy: agentcore.ToolPolicy{
        Timeout:     30 * time.Second,
        MaxAttempts: 2,
    },
    MaxToolConcurrency: 4,
    ContextPolicy: agentcore.ContextPolicy{
        MaxTokens:          100_000,
        ReserveTokens:      8_000,
        MaxToolResultBytes: 64 * 1024,
        Compact:            summarizeContext,
    },
})
```

When no `ContextPolicy.Compact` function is supplied, the core keeps the newest
valid message suffix. Production applications should supply a model-aware token
estimator and summarizer when exact context behavior matters.

`NewSummaryCompactor` provides a ready-made model-backed compactor that keeps a
recent suffix and replaces older history with a summary. Tool timeouts are
cooperative: implementations must observe `ctx` because Go cannot forcibly stop
an arbitrary goroutine.

`PrepareNextTurn` may set `TurnContext.Next` to switch the model, tools, system
prompt, execution mode, or model options without mutating the reusable Agent.

## Event order

A run with one tool call emits:

```text
agent_start
turn_start
message_start / message_end        user prompt
message_start / message_update... / message_end
tool_execution_start / update... / end
message_start / message_end        tool result
turn_end
turn_start
message_start / message_update... / message_end
turn_end
agent_end
```

Parallel tools emit completion events in completion order, while tool-result
messages are appended to state in the model's original call order.

See [PI_CONFORMANCE.md](PI_CONFORMANCE.md) for the tested behavioral boundary.
