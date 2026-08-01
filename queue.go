package agentcore

type messageQueue interface {
	takeSteering() []Message
	takeFollowUp() []Message
}
