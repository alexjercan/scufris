package tools

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type HomeParameters struct{}

func (p *HomeParameters) Validate(tool tool.Tool) error {
	return nil
}

func (p *HomeParameters) String() string {
	return ""
}

type HomeToolResponse struct {
	Home string `json:"home" jsonschema:"title=home,description=The current user's home directory."`
}

func (r *HomeToolResponse) String() string {
	return fmt.Sprintf("%s", r.Home)
}

func (r *HomeToolResponse) Image() uuid.UUID {
	return uuid.Nil
}

type HomeTool struct {
	logger *slog.Logger
}

func NewOsHomeTool() tool.Tool {
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

func (t *HomeTool) Call(ctx context.Context, params tool.ToolParameters) (tool.ToolResponse, error) {
	t.logger.Debug("HomeTool.Call called", slog.String("name", t.Name()))

	home, err := os.UserHomeDir()
	if err != nil {
		return nil, &scufris.Error{
			Code:    "OS_HOME_ERROR",
			Message: fmt.Sprintf("failed to get home directory: %v", err),
			Err:    fmt.Errorf("failed to get home directory: %w", err),
		}
	}

	t.logger.Debug("HomeTool.Call completed", slog.String("home", home))

	return &HomeToolResponse{Home: home}, nil
}
