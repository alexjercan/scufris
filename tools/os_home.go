package tools

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"reflect"

	"github.com/alexjercan/scufris/internal/observer"
)

type HomeParameters struct{}

func (p *HomeParameters) Validate(tool Tool) error {
	return nil
}

type HomeTool struct {
	logger *slog.Logger
}

func NewHomeTool() Tool {
	return &HomeTool{
		logger: slog.Default(),
	}
}

func (t *HomeTool) Name() string {
	return "os-home"
}

func (t *HomeTool) Description() string {
	return "Get the current user's home directory."
}

func (t *HomeTool) Parameters() reflect.Type {
	return reflect.TypeOf(HomeParameters{})
}

func (t *HomeTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	t.logger.Debug("HomeTool.Call called", slog.String("name", t.Name()))

	observer.OnStart(ctx)
	err := observer.OnToken(ctx, "I need to get the current user's home directory.")
	if err != nil {
		return nil, err
	}
	observer.OnEnd(ctx)

	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get home directory: %w", err)
	}

	t.logger.Debug("HomeTool.Call completed", slog.String("home", home))

	return home, nil
}
