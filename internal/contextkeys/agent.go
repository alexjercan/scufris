package contextkeys

import "context"

type agentNameKeyType struct{}

var agentNameKey = agentNameKeyType{}

func WithAgentName(ctx context.Context, name string) context.Context {
	return context.WithValue(ctx, agentNameKey, name)
}

func AgentName(ctx context.Context) (string, bool) {
	name, ok := ctx.Value(agentNameKey).(string)
	return name, ok
}
