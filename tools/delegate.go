package tools

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/verbose"
)

type AgentLike interface {
	Name() string
	Description() string

	Chat(ctx context.Context, prompt string) (string, error)
}

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
	agent  AgentLike
	logger *slog.Logger
}

func NewDelegateTool(agent AgentLike) Tool {
	return &DelegateTool{
		agent:  agent,
		logger: slog.Default(),
	}
}

func (t *DelegateTool) Parameters() reflect.Type {
	return reflect.TypeOf(DelegateToolParameters{})
}

func (t *DelegateTool) Name() string {
	return fmt.Sprintf("delegate_to_%s", t.agent.Name())
}

func (t *DelegateTool) Description() string {
	return t.agent.Description()
}

func (t *DelegateTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	prompt := params.(*DelegateToolParameters).Prompt

	if name, ok := contextkeys.AgentName(ctx); ok {
		verbose.Say(name, fmt.Sprintf("I need to check with %s: %s", t.agent.Name(), prompt))
	}

	return t.agent.Chat(ctx, prompt)
}
