package anthropic

import (
	"context"
	"errors"
	"fmt"
	"net/http"
)

// Error is a structured Anthropic transport or API error.
type Error struct {
	Operation  string
	StatusCode int
	Type       string
	Message    string
	Retryable  bool
	Err        error
}

func (e *Error) Error() string {
	if e == nil {
		return "provider/anthropic: <nil>"
	}
	result := "provider/anthropic"
	if e.Operation != "" {
		result += ": " + e.Operation
	}
	if e.StatusCode != 0 {
		result += fmt.Sprintf(": HTTP %d", e.StatusCode)
	}
	if e.Type != "" {
		result += " (" + e.Type + ")"
	}
	if e.Message != "" {
		result += ": " + e.Message
	} else if e.Err != nil {
		result += ": " + e.Err.Error()
	}
	return result
}
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// IsRetryable reports whether an Anthropic error is transient.
func IsRetryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var target *Error
	return errors.As(err, &target) && target.Retryable
}

func retryableStatus(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusConflict || status == http.StatusTooManyRequests || status >= 500
}
