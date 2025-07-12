package contextkeys

import "context"

// private key type to avoid collision
type agentNameKeyType struct{}

var agentNameKey = agentNameKeyType{}

// Setter
func WithAgentName(ctx context.Context, name string) context.Context {
	return context.WithValue(ctx, agentNameKey, name)
}

// Getter
func AgentName(ctx context.Context) (string, bool) {
	name, ok := ctx.Value(agentNameKey).(string)
	return name, ok
}
