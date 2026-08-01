package agentcore

import "time"

func normalizeRetryPolicy(policy RetryPolicy) RetryPolicy {
	if policy.MaxAttempts <= 0 {
		policy.MaxAttempts = 1
	}
	if policy.Multiplier < 1 {
		policy.Multiplier = 2
	}
	if policy.MaxAttempts > 1 && policy.InitialDelay <= 0 {
		policy.InitialDelay = 100 * time.Millisecond
	}
	if policy.MaxDelay <= 0 {
		policy.MaxDelay = 2 * time.Second
	}
	return policy
}

func normalizeToolPolicy(policy ToolPolicy) ToolPolicy {
	if policy.MaxAttempts <= 0 {
		policy.MaxAttempts = 1
	}
	if policy.MaxAttempts > 1 && policy.RetryDelay <= 0 {
		policy.RetryDelay = 50 * time.Millisecond
	}
	return policy
}

func mergeToolPolicy(base, override ToolPolicy) ToolPolicy {
	if override.Timeout != 0 {
		base.Timeout = override.Timeout
	}
	if override.MaxAttempts != 0 {
		base.MaxAttempts = override.MaxAttempts
	}
	if override.RetryDelay != 0 {
		base.RetryDelay = override.RetryDelay
	}
	if override.ShouldRetry != nil {
		base.ShouldRetry = override.ShouldRetry
	}
	if override.DisablePanicRecovery {
		base.DisablePanicRecovery = true
	}
	return normalizeToolPolicy(base)
}

func retryDelay(policy RetryPolicy, failedAttempt int) time.Duration {
	delay := policy.InitialDelay
	for attempt := 1; attempt < failedAttempt; attempt++ {
		delay = time.Duration(float64(delay) * policy.Multiplier)
		if delay >= policy.MaxDelay {
			return policy.MaxDelay
		}
	}
	if delay > policy.MaxDelay {
		return policy.MaxDelay
	}
	return delay
}
