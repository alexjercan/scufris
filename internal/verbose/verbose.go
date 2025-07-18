package verbose

import (
	"context"
	"fmt"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/pretty"
	"github.com/alexjercan/scufris/internal/registry"
)

type verboseObserver struct {
}

func NewVerboseObserver() observer.Observer {
	return &verboseObserver{}
}

func (v *verboseObserver) OnStart(ctx context.Context) error {
	if name, ok := contextkeys.AgentName(ctx); ok {
		return pretty.OnStart(name)
	}

	return nil
}

func (v *verboseObserver) OnToken(ctx context.Context, token string) error {
	return pretty.OnToken(token)
}

func (v *verboseObserver) OnEnd(ctx context.Context) error {
	return pretty.OnEnd()
}

func (v *verboseObserver) OnError(ctx context.Context, err error) error {
	return pretty.OnError(err)
}

func (v *verboseObserver) OnImage(ctx context.Context, imageId string) error {
	if img, ok := registry.GetImage(ctx, imageId); ok {
		return pretty.OnImage(img)
	} else {
		return &scufris.Error{
			Code:    "IMAGE_NOT_FOUND",
			Message: fmt.Sprintf("image with id %s not found in registry", imageId),
			Err:     fmt.Errorf("image with id %s not found in registry", imageId),
		}
	}
}

func (v *verboseObserver) OnToolCall(ctx context.Context, toolName string, args any) error {
	return pretty.OnToolCall(toolName, args)
}

func (v *verboseObserver) OnToolCallEnd(ctx context.Context, toolName string, result any) error {
	return pretty.OnToolCallEnd(toolName, result)
}
