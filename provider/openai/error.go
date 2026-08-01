package openai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// Error is a bounded, structured provider or transport failure.
type Error struct {
	Operation  string
	StatusCode int
	Code       string
	Type       string
	Body       string
	RetryAfter time.Duration
	Retryable  bool
	Err        error
}

func (e *Error) Error() string {
	if e == nil {
		return "provider/openai: <nil>"
	}
	prefix := "provider/openai"
	if e.Operation != "" {
		prefix += ": " + e.Operation
	}
	if e.StatusCode != 0 {
		prefix += fmt.Sprintf(": HTTP %d", e.StatusCode)
	}
	if e.Code != "" {
		prefix += " (" + e.Code + ")"
	}
	if e.Body != "" {
		return prefix + ": " + e.Body
	}
	if e.Err != nil {
		return prefix + ": " + e.Err.Error()
	}
	return prefix
}

func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// Temporary supports callers that use the conventional temporary-error
// interface. Prefer IsRetryable for agentcore.RetryPolicy.ShouldRetry.
func (e *Error) Temporary() bool { return e != nil && e.Retryable }

// IsRetryable reports whether err represents a transient provider failure.
// It is intended for agentcore.RetryPolicy.ShouldRetry.
func IsRetryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var providerError *Error
	if errors.As(err, &providerError) {
		return providerError.Retryable
	}
	return false
}

func retryableStatus(status int) bool {
	return status == http.StatusRequestTimeout ||
		status == http.StatusConflict ||
		status == http.StatusTooEarly ||
		status == http.StatusTooManyRequests ||
		status >= http.StatusInternalServerError
}
