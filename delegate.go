package scufris

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris/tools"
)

type DelegateToolParameters struct {
	Prompt string `json:"prompt" jsonschema:"title=prompt,description=The prompt to use for the delegate agent."`
}

func (p *DelegateToolParameters) Validate() error {
	if p.Prompt == "" {
		return fmt.Errorf("prompt cannot be empty")
	}
	return nil
}

type DelegateTool struct {
	Params DelegateToolParameters

	agent  *Agent
	logger *slog.Logger
}

func NewDelegateTool(agent *Agent) tools.Tool {
	return &DelegateTool{
		agent:  agent,
		logger: slog.Default(),
	}
}

func (t *DelegateTool) Name() string {
	return fmt.Sprintf("delegate_to_%s", t.agent.Name)
}

func (t *DelegateTool) Description() string {
	return t.agent.Description
}

func (t *DelegateTool) Call(ctx context.Context) (any, error) {
	return t.agent.Chat(ctx, t.Params.Prompt)
}
