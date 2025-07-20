package verbose

import (
	"context"

	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/pretty"
)

type verboseObserver struct {
}

func NewVerboseObserver() observer.Observer {
	return &verboseObserver{}
}

func (o *verboseObserver) OnUser(ctx context.Context, message string) error {
	return nil
}

func (o *verboseObserver) OnStart(ctx context.Context) error {
	if name, ok := contextkeys.AgentName(ctx); ok {
		return pretty.OnStart(name)
	}

	return nil
}

func (o *verboseObserver) OnToken(ctx context.Context, token string) error {
	return pretty.OnToken(token)
}

func (o *verboseObserver) OnEnd(ctx context.Context) error {
	return pretty.OnEnd()
}

func (o *verboseObserver) OnError(ctx context.Context, err error) error {
	return pretty.OnError(err)
}

func (o *verboseObserver) OnImage(ctx context.Context, image string) error {
	return pretty.OnImage(image)
}

func (o *verboseObserver) OnToolCall(ctx context.Context, toolName string, args any) error {
	return pretty.OnToolCall(toolName, args)
}

func (o *verboseObserver) OnToolCallEnd(ctx context.Context, toolName string, result any) error {
	return pretty.OnToolCallEnd(toolName, result)
}
