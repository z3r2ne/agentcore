// Package sqlitestore persists agentcore session snapshots in SQLite.
//
// The store uses one atomically replaced JSON payload per session. This keeps
// provider data and future message fields lossless while still exposing useful
// SQLite metadata such as message count and update time.
package sqlitestore
