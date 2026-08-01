// Package agentcore implements a provider-neutral, Pi-style streaming agent
// loop with composable tools, skills, lifecycle interceptors, and sessions.
//
// The package intentionally owns only the execution semantics: model turns,
// tool calls, event delivery, cancellation, and in-memory conversation state.
// Persistence is optional through SessionStore adapters; user interfaces,
// authentication, and provider selection belong to callers or adapters.
package agentcore
