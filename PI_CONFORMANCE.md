# Pi core conformance

The behavioral reference is `@earendil-works/pi-agent-core` 0.82.1 at Pi
commit `027a5847901b5dde30270abaa1041046cd2b4b55e`.

The conformance tests lock down the observable core behavior rather than Pi's
TypeScript API surface:

| Pi behavior | Go evidence |
|---|---|
| Agent, turn, message, and tool event order | `TestPiConformanceToolLoopAndEventOrder` |
| Parallel completion events with source-ordered tool-result messages | `TestPiConformanceParallelCompletionAndPersistenceOrder` |
| Tool calls from length-truncated responses are rejected | `TestLengthStopRejectsToolWithoutExecution` |
| Steering is delivered before follow-up messages | `TestPiConformanceSteeringThenFollowUpDelivery` |
| One-at-a-time queue delivery | `TestPiConformanceSteeringThenFollowUpDelivery` |
| All-at-once queue delivery | `TestSessionDeliveryAllInjectsEachQueueAsOneTurn` |
| Abort ends with an aborted result and terminal event | `TestToolCancellationEndsRunAsAborted`, `TestSessionAbortCancelsActiveModelAndSettles` |
| Awaited listeners form an event-order barrier | synchronous `EventSink` delivery and race tests |

Intentional API differences:

- Go uses typed callbacks or `EventStream` instead of a TypeScript event emitter.
- Model deltas are normalized as `ModelChunk` rather than Pi provider events.
- State is explicitly passed to stateless `Agent` calls or owned by `Session`.
- Provider-specific messages are stored in serializable `ProviderData`.

RPC commands, TUI events, session files, authentication, and Pi extensions are
outside this package's conformance boundary.
