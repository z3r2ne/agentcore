# agentcore 中文使用指南

本文面向希望把 `agentcore` 作为 Go 库嵌入自己项目的开发者。示例基于 `v0.2.1`。

## 1. 它负责什么

`agentcore` 只负责 Agent 的执行语义：

```text
用户消息 → Model 流式响应 → Tool Call → Tool Result → Model → 最终响应
```

它提供：

- Provider 无关的流式 Agent Loop；
- 自定义 Model 与 Tool；
- Tool 参数 JSON Schema 校验、超时、重试和并发控制；
- 可动态加载指令、工具和拦截器的 Skill；
- Model/Tool 调用前后拦截；
- Session、Steer、FollowUp、Abort 和事件订阅；
- 上下文预算、压缩和 Token/Cost 统计；
- 可选的 SQLite Session 持久化；
- Eino Model 与 Tool 适配器。

它不负责 HTTP API、任务调度、用户系统、权限数据库、MCP 管理、TUI 或具体 Provider 登录。宿主项目应把这些能力转换成 `Model`、`Tool`、`Skill`、`Interceptor` 或 `SessionStore`。

## 2. 安装

公开仓库直接安装：

```bash
go get github.com/z3r2ne/agentcore@v0.2.1
```

如果仓库保持私有，先让 Go 跳过公共代理和校验服务，并让 Git 使用已有 SSH 权限：

```bash
go env -w GOPRIVATE=github.com/z3r2ne/*
git config --global url."ssh://git@ssh.github.com:443/".insteadOf https://github.com/
go get github.com/z3r2ne/agentcore@v0.2.1
```

CI 中应使用只读 Deploy Key、GitHub App Token 或最小权限凭据，不要把个人 Token 写进 `go.mod`、源码或镜像。

## 3. 最小 Agent

首先准备一个实现了 `agentcore.Model` 的模型适配器，然后创建 Agent：

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/z3r2ne/agentcore"
)

func run(ctx context.Context, model agentcore.Model) {
    agent, err := agentcore.New(agentcore.Config{
        Model:        model,
        SystemPrompt: "你是一个简洁、可靠的助手。",
        MaxTurns:     32,
    })
    if err != nil {
        log.Fatal(err)
    }

    result, err := agent.Prompt(
        ctx,
        agentcore.State{},
        []agentcore.Message{
            agentcore.TextMessage(agentcore.RoleUser, "解释 Go context 的用途"),
        },
        nil,
    )
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result.State.Messages[len(result.State.Messages)-1].Text())
}
```

`Agent` 配置在创建后不可变，可以并发复用；单次会话状态放在传入的 `State` 或 `Session` 中。

## 4. 接入模型

模型适配器只需实现：

```go
type Model interface {
    Stream(context.Context, ModelRequest) (ModelStream, error)
}

type ModelStream interface {
    Recv() (ModelChunk, error)
    Close() error
}
```

适配器负责：

1. 把 `ModelRequest.SystemPrompt`、`Messages`、`Tools` 和 `Options` 转换成 Provider 请求；
2. 把流式文本、思考内容和工具调用转换成 `ModelChunk`；
3. 流结束时返回 `io.EOF`；
4. 把 Provider 的 Token 用量写入 `ModelChunk.Usage`；
5. 在 `Close` 中释放网络流。

### 4.1 OpenAI-compatible Chat Completions

仓库内置了可选的具体 Provider，它位于独立子包，不会把 OpenAI 配置或 HTTP 行为带入根包：

```go
import (
    "net/http"
    "os"
    "time"

    "github.com/z3r2ne/agentcore"
    openaiprovider "github.com/z3r2ne/agentcore/provider/openai"
)

model, err := openaiprovider.New(openaiprovider.Config{
    Model:   "gpt-4.1",
    APIKey:  os.Getenv("OPENAI_API_KEY"),
    BaseURL: "https://api.openai.com/v1",

    // OpenAI-compatible 服务可以替换 BaseURL，并加入自定义 Header。
    Headers: map[string]string{
        "OpenAI-Organization": "org_example",
    },

    // 可选。可以传入设置过代理、连接池、TLS 或超时的客户端。
    HTTPClient: &http.Client{Timeout: 2 * time.Minute},

    // 以下为不可信响应的读取边界；0 使用安全默认值。
    MaxResponseBodyBytes: 64 << 20,
    MaxErrorBodyBytes:    1 << 20,
    MaxSSEEventBytes:     4 << 20,
})
if err != nil {
    return err
}

agent, err := agentcore.New(agentcore.Config{
    Model: model,
    ModelRetry: agentcore.RetryPolicy{
        MaxAttempts:  3,
        InitialDelay: 250 * time.Millisecond,
        MaxDelay:     2 * time.Second,
        ShouldRetry:  openaiprovider.IsRetryable,
    },
})
```

也可以把模型名与连接配置分开：

```go
model, err := openaiprovider.NewModel(openaiprovider.Config{
    BaseURL: "https://your-compatible-service.example/v1",
    APIKey:  os.Getenv("MODEL_API_KEY"),
}, "your-model")
```

该 Provider 支持：

- 标准 SSE 事件，包括多行 `data:`、注释和 `[DONE]`；
- 一个响应中的多个并行 `tool_calls` 及分片参数拼接；
- `content`、`reasoning_content`、`usage` 和常见缓存 Token 字段；
- 把兼容服务的扩展消息字段保存为可序列化 `ProviderData`，工具执行后的下一轮会带回；
- 自定义 `BaseURL`、API Key、单值/多值 Header 与 `http.Client`；
- 请求阶段和流读取阶段的 `context.Context` 取消；
- 错误响应体、单个 SSE Event、完整流响应体三层大小限制；
- 结构化 `*openaiprovider.Error`，包含状态码、错误码、类型、`Retry-After` 与是否可重试。

`Config.Header` 是 `http.Header`，适合多值 Header，并在 `Headers` 之后应用。自定义 `Authorization` 会覆盖由 `APIKey` 生成的 Bearer Header。构造时会复制 Header，之后修改调用方的 map 不会改变 Model。

`agentcore.Config.ModelOptions` 会作为 Chat Completions 顶层参数传给 Provider，例如：

```go
agent, err := agentcore.New(agentcore.Config{
    Model: model,
    ModelOptions: map[string]any{
        "temperature":      0,
        "max_tokens":       4096,
        "reasoning_effort": "medium",
    },
})
```

`model`、`messages`、`tools`、`stream`、`stream_options` 和 `n` 由适配器管理，不能通过 `ModelOptions` 覆盖。适配器固定请求单个候选，避免多个 choice 混入同一个 Agent 消息。

非 2xx 响应和流中错误都可以用 `errors.As` 读取：

```go
var providerErr *openaiprovider.Error
if errors.As(err, &providerErr) {
    log.Printf("status=%d code=%s retryAfter=%s", providerErr.StatusCode, providerErr.Code, providerErr.RetryAfter)
}
```

如果不设置 `ShouldRetry`，`agentcore` 会重试所有模型传输/流错误；生产环境推荐使用 `openaiprovider.IsRetryable`，从而只重试 408、409、425、429、5xx、网络错误和流中临时错误，不重试普通 4xx 或 Context 取消。

### 4.2 Eino Adapter

如果项目已经使用 CloudWeGo Eino，可以直接使用内置适配器：

```go
import "github.com/z3r2ne/agentcore/einoadapter"

coreModel := einoadapter.Model{ChatModel: einoChatModel}
agent, err := agentcore.New(agentcore.Config{Model: coreModel})
```

Provider 特有的 reasoning signature、消息元数据等可以放入 `ProviderData`。Eino Adapter 会在工具循环和 Session JSON 序列化之间保留这些数据。

## 5. 自定义 Tool

最简单的方式是使用 `FuncTool`：

```go
type weatherArgs struct {
    City string `json:"city"`
}

weatherTool := agentcore.FuncTool{
    ToolDefinition: agentcore.ToolDefinition{
        Name:        "weather",
        Description: "查询指定城市天气",
        Parameters: json.RawMessage(`{
            "type": "object",
            "properties": {
                "city": {"type": "string", "minLength": 1}
            },
            "required": ["city"],
            "additionalProperties": false
        }`),
    },
    Policy: agentcore.ToolPolicy{
        Timeout:     15 * time.Second,
        MaxAttempts: 2,
        RetryDelay:  300 * time.Millisecond,
    },
    ExecuteFunc: func(
        ctx context.Context,
        raw json.RawMessage,
        update agentcore.ToolUpdateSink,
    ) (agentcore.ToolResult, error) {
        var args weatherArgs
        if err := json.Unmarshal(raw, &args); err != nil {
            return agentcore.ToolResult{}, err
        }

        _ = update(agentcore.TextToolResult("正在连接天气服务"))
        temperature, err := queryWeather(ctx, args.City)
        if err != nil {
            return agentcore.ToolResult{}, err
        }
        return agentcore.TextToolResult(
            fmt.Sprintf("%s 当前温度 %.1f°C", args.City, temperature),
        ), nil
    },
}
```

注意事项：

- JSON Schema 会在工具执行前自动编译和校验；
- `ExecuteFunc` 必须响应 `ctx.Done()`，Go 无法强制终止无视 Context 的 goroutine；
- `update` 用于流式报告中间状态，不代表最终 Tool Result；
- 工具 panic 默认会被转成错误结果；
- 副作用工具应使用稳定幂等键。

工具执行期间可以读取稳定调用信息：

```go
invocation, ok := agentcore.ToolInvocationFromContext(ctx)
if ok {
    fmt.Println(invocation.Turn)
    fmt.Println(invocation.Attempt)
    fmt.Println(invocation.Call.ID)
}
```

可以用 `Call.ID + 业务 Execution ID` 作为数据库写入或外部 API 调用的幂等键。

## 6. Skill

Skill 是构建 Agent 时动态加载的一组可信指令、工具和拦截器。

静态 Skill：

```go
reviewSkill := agentcore.FuncSkill{
    SkillDefinition: agentcore.SkillDefinition{
        Name:        "code-review",
        Version:     "1",
        Description: "代码正确性与安全性检查",
    },
    Content: agentcore.SkillContent{
        Instructions: "检查正确性、安全边界和测试证据，结论必须引用具体代码。",
        Tools:        []agentcore.Tool{diffTool, testTool},
    },
}
```

动态 Skill：

```go
reviewSkill := agentcore.FuncSkill{
    SkillDefinition: agentcore.SkillDefinition{Name: "code-review", Version: "2"},
    LoadFunc: func(ctx context.Context) (agentcore.SkillContent, error) {
        instructions, err := os.ReadFile("skills/code-review/SKILL.md")
        if err != nil {
            return agentcore.SkillContent{}, err
        }
        return agentcore.SkillContent{
            Instructions: string(instructions),
            Tools:        []agentcore.Tool{diffTool, testTool},
            Interceptors: []agentcore.Interceptor{auditInterceptor},
        }, nil
    },
}
```

使用 Builder 组装：

```go
agent, err := agentcore.NewBuilder(model).
    SystemPrompt("你是软件工程 Agent。" ).
    Tools(projectSearchTool).
    Skills(reviewSkill).
    Build(ctx)
```

同一个 Builder 中 Skill 名称不能重复，Skill 提供的 Tool 名称也必须全局唯一。Skill 指令会进入 System Prompt，因此只能加载可信内容；用户上传文本应作为普通 Message 或 Tool Result，而不是 Skill 指令。

## 7. Tool 调用前后拦截

`Interceptor` 适合实现审批、权限、审计、缓存、参数改写、结果过滤和计费。

```go
approval := agentcore.InterceptorFuncs{
    Name: "tool-approval",
    BeforeTool: func(
        ctx context.Context,
        call agentcore.ToolCallContext,
    ) (agentcore.ToolCallDecision, error) {
        if call.Call.Name == "shell" && !approved(ctx, call.Call) {
            return agentcore.ToolCallDecision{
                Block:  true,
                Reason: "此命令需要人工审批",
            }, nil
        }

        // 也可以返回 Arguments，在 JSON Schema 校验前重写参数。
        return agentcore.ToolCallDecision{}, nil
    },
    AfterTool: func(
        ctx context.Context,
        call agentcore.ToolCallContext,
        result *agentcore.ToolResult,
    ) error {
        audit(ctx, call.Call, call.Executed, call.Attempts, *result)

        // 可以在交给模型之前脱敏结果。
        redactToolResult(result)
        return nil
    },
}
```

注册多个拦截器时：

```text
Before: A → B → Tool
After:  Tool → B → A
```

后置拦截器也会收到被阻止、工具不存在或参数校验失败的调用：

- `call.Executed == false`：工具没有真正执行；
- `call.Attempts`：实际尝试次数；
- `result.IsError`：返回给模型的结果是否为错误。

拦截器挂在可并发复用的 Agent 上，且并行工具会并发触发拦截器。拦截器若保存可变状态，必须自行加锁；执行级数据优先放进 `context.Context`。

## 8. Model 和 Turn 拦截

同一个 `InterceptorFuncs` 还可以处理 Model 与 Turn 生命周期：

```go
runtimePolicy := agentcore.InterceptorFuncs{
    Name: "runtime-policy",
    BeforeModel: func(ctx context.Context, request *agentcore.ModelRequest) error {
        if request.Options == nil {
            request.Options = make(map[string]any)
        }
        request.Options["trace_id"] = traceIDFromContext(ctx)
        return nil
    },
    AfterModel: func(ctx context.Context, message *agentcore.Message) error {
        inspectModelOutput(message)
        return nil
    },
    PrepareTurn: func(ctx context.Context, turn *agentcore.TurnContext) error {
        if shouldUseCarefulModel(turn) {
            turn.Next = &agentcore.NextTurnConfig{Model: carefulModel}
        }
        return nil
    },
    Stop: func(ctx context.Context, turn agentcore.TurnContext) bool {
        return applicationBudgetExceeded(ctx, turn.Usage)
    },
}
```

可用扩展接口包括：

- `BeforeModelCallInterceptor`；
- `AfterModelCallInterceptor`；
- `BeforeToolCallInterceptor`；
- `AfterToolCallInterceptor`；
- `PrepareNextTurnInterceptor`；
- `ShouldStopInterceptor`。

`Hooks` 仍适合一个调用方直接设置少量回调；多个独立模块共同扩展 Agent 时优先使用 `Interceptors`。

## 9. 推荐 Builder 配置

```go
agent, err := agentcore.NewBuilder(model).
    SystemPrompt("你是生产环境中的任务 Agent。" ).
    Tools(baseTools...).
    Skills(selectedSkills...).
    Use(approval, audit, metrics).
    Configure(func(config *agentcore.Config) {
        config.MaxTurns = 64
        config.ToolExecution = agentcore.ToolExecutionParallel
        config.MaxToolConcurrency = 4

        config.ModelRetry = agentcore.RetryPolicy{
            MaxAttempts:  3,
            InitialDelay: 200 * time.Millisecond,
            MaxDelay:     2 * time.Second,
        }
        config.ToolPolicy = agentcore.ToolPolicy{
            Timeout:     30 * time.Second,
            MaxAttempts: 2,
            RetryDelay:  200 * time.Millisecond,
        }
        config.ContextPolicy = agentcore.ContextPolicy{
            MaxTokens:          100_000,
            ReserveTokens:      8_000,
            MaxToolResultBytes: 64 * 1024,
        }
    }).
    Build(ctx)
```

`Builder.Build` 是 Skill 加载和最终配置校验边界。建议为每种执行能力组合构建 Agent，而不是在运行中修改共享配置。

## 10. 同步与流式执行

同步回调：

```go
result, err := agent.Prompt(ctx, state, prompts, func(
    ctx context.Context,
    event agentcore.Event,
) error {
    publishEvent(event)
    return nil
})
```

异步事件流：

```go
stream := agent.Stream(ctx, state, prompts)
for {
    event, ok := stream.Next()
    if !ok {
        break
    }
    if event.Type == agentcore.EventMessageUpdate && event.Delta != nil {
        fmt.Print(event.Delta.TextDelta)
    }
}
result, err := stream.Result()
```

`EventStream` 使用内存中的无界事件队列。消费者短暂变慢不会阻塞模型，但长期不读取可能增加内存；生产系统应持续消费并转发到自己的有界消息系统。

Tool 参数生成过程不会丢失。`message_update` 中的
`event.Delta.ToolCallDeltas[].ArgumentsDelta` 是任意文本片段，可能只是 `{`
或 `"path":`，调用方可以按 `Index` 累积并显示打字机效果。此时它不是
JSON，不能按 `json.RawMessage` 处理。只有 `message_end` 的
`event.Message.ToolCalls()` 以及后续 `tool_execution_start` 才包含最终、完整且
合法的 `Arguments`。

## 11. Session、Steer 与 FollowUp

`Session` 持有持续会话状态，同一时间只允许一个 Prompt 正在执行：

```go
session, err := agentcore.NewSession(
    agent,
    agentcore.State{},
    agentcore.SessionOptions{
        SteeringMode: agentcore.DeliveryAll,
        FollowUpMode: agentcore.DeliveryAll,
    },
)

run := session.Stream(ctx, []agentcore.Message{
    agentcore.TextMessage(agentcore.RoleUser, "开始分析任务"),
})

err = session.Steer(
    agentcore.TextMessage(agentcore.RoleUser, "优先检查权限边界"),
)

err = session.FollowUp(
    agentcore.TextMessage(agentcore.RoleUser, "完成后给出风险摘要"),
)

result, err := run.Result()
```

区别：

- `Steer`：当前 Model Turn 和工具调用完成后尽快注入；
- `FollowUp`：Agent 原本准备停止时注入；
- `Abort`：取消当前 Model Stream 和遵守 Context 的 Tool；
- `WaitForIdle`：等待当前运行结束；
- `Status`：读取流式消息、待执行 Tool、队列和 Usage；
- `State`：读取已经提交的会话状态；
- `Subscribe`：长期订阅 Session 事件。

在 Session 空闲时调用 `Steer`、`FollowUp` 或 `Abort` 会返回 `ErrSessionIdle`。

## 12. SQLite 持久化

```go
import "github.com/z3r2ne/agentcore/sqlitestore"

store, err := sqlitestore.Open("./data/agentcore.db")
if err != nil {
    log.Fatal(err)
}
defer store.Close()

session, err := agentcore.NewSession(
    agent,
    agentcore.State{},
    agentcore.SessionOptions{
        Store:     store,
        SessionID: "task-123-agent-7",
    },
)
```

每次运行完成以及 Steer/FollowUp 入队后都会保存原子快照。进程重启后恢复：

```go
session, err := store.RestoreSession(
    ctx,
    "task-123-agent-7",
    agent,
    agentcore.SessionOptions{},
)
```

还可以使用 `ListSessions`、`LoadSession`、`SaveSession` 和 `DeleteSession` 管理生命周期。SQLite 文件默认启用 WAL、busy timeout，并设置为 `0600`。

Session Store 只保存消息、Usage、投递队列和 ProviderData，不保存运行中的网络连接、Tool 实例、订阅者或 Context。

## 13. 上下文管理

```go
compactor, err := agentcore.NewSummaryCompactor(agentcore.SummaryCompactorConfig{
    Model:        summaryModel,
    KeepRecent:   8,
    SystemPrompt: "压缩对话，保留约束、决策、路径、错误和未完成事项。",
})

agent, err := agentcore.New(agentcore.Config{
    Model: model,
    ContextPolicy: agentcore.ContextPolicy{
        MaxTokens:          100_000,
        ReserveTokens:      8_000,
        MaxToolResultBytes: 64 * 1024,
        EstimateTokens:     modelTokenizer,
        Compact:            compactor,
    },
})
```

没有自定义 `EstimateTokens` 时使用 Provider 无关估算；生产环境需要精确预算时应接入目标模型 tokenizer。没有 `Compact` 时，核心会保留最新的合法消息后缀，并避免留下孤立 Tool Result。

## 14. 事件与可观测性

常用事件：

```text
agent_start / agent_end
turn_start / turn_end
message_start / message_update / message_end
tool_execution_start / update / end
auto_retry_start / auto_retry_end
context_compaction_start / context_compaction_end
```

建议在宿主项目中给 Context 注入 `taskId`、`executionId`、`traceId` 等关联信息，并在 EventSink、Interceptor 和 Tool 中统一读取。不要直接记录密钥、完整 Authorization Header 或未经脱敏的 Tool Result。

## 15. 错误、取消与重试

- Model 传输或流错误可以使用 `ModelRetry`；
- Tool 重试由全局 `ToolPolicy` 与 Tool 自己的 Policy 合并；
- JSON Schema 错误、未知 Tool 和被拦截调用会成为 Tool Error Result，让模型有机会修正；
- EventSink 返回错误会停止当前运行；
- `context.Canceled` 和 `context.DeadlineExceeded` 对应 `StopReasonAborted`；
- 达到最大轮数返回 `ErrMaxTurns` 与 `StopReasonMaxTurns`；
- `LifecycleError` 会标明出错的 Interceptor 与阶段。

是否重试有副作用的 Tool 必须谨慎。只有在工具具备幂等语义时，才应配置自动重试。

## 16. 并发模型

- 创建完成的 `Agent` 可以并发复用；
- `Builder` 是可变装配器，不应并发修改；
- 一个 `Session` 同时只能运行一个 Prompt；
- 一个 Turn 中 Tool 默认可并行执行；
- 使用 `ToolExecutionSequential` 或 Tool 的 `ExecutionMode` 强制顺序；
- `MaxToolConcurrency` 限制单批工具并发；
- Model、Tool、Skill Loader、Interceptor、EventSink 和 SessionStore 都应遵守 Context，并明确自己的线程安全边界。

## 17. 生产检查清单

- 固定明确版本，例如 `v0.2.1`，不要依赖浮动 `main`；
- Model 凭据由宿主 Secret 系统管理；
- 为危险 Tool 添加审批/权限 Interceptor；
- 为副作用 Tool 设计幂等键；
- 配置 Model/Tool 超时、最大轮数和上下文预算；
- 限制 Tool Result 大小并对日志脱敏；
- 持续消费 EventStream；
- 使用 SessionStore 时设计 Session ID 的租户和任务隔离；
- 对自定义 Tool、Skill、Interceptor 运行 `go test -race`；
- 在升级版本前阅读 Release Notes，并运行宿主项目完整测试。

## 18. 推荐的项目边界

```text
your-project/
├── models/          Provider → agentcore.Model
├── tools/           业务工具
├── skills/          Skill 加载与可信指令
├── interceptors/    审批、权限、审计、指标
├── sessions/        SessionStore 与恢复策略
├── orchestration/   任务、队列、父子 Agent、定时唤醒
└── cmd/             HTTP、CLI 或 Worker 组合根
```

不要把业务任务状态机放入 `agentcore`。核心库只运行一个 Agent；任务编排、队列、协同模式和多 Agent 生命周期应由宿主项目管理。

## 19. 更多资料

- [英文 README](../README.md)
- [Pi 行为一致性边界](../PI_CONFORMANCE.md)
- [v0.2.1 Release](https://github.com/z3r2ne/agentcore/releases/tag/v0.2.1)
