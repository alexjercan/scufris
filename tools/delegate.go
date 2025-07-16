package tools

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/verbose"
	"github.com/alexjercan/scufris/llm"
)

type AgentLike interface {
	Name() string
	Description() string

	Chat(ctx context.Context, message llm.Message) (string, error)
}

type DelegateToolParameters struct {
	Prompt   string   `json:"prompt" jsonschema:"title=prompt,description=The prompt to use for the delegate agent."`
	ImageIds []string `json:"image_ids,omitempty" jsonschema:"title=image_ids,description=The image ids to use with the delegate agent. SHOULD ONLY BE USED BY VISION AGENTS, otherwise it will be ignored."`
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
	imageIds := params.(*DelegateToolParameters).ImageIds

	if name, ok := contextkeys.AgentName(ctx); ok {
		if len(imageIds) == 0 {
			verbose.Say(name, fmt.Sprintf("I need to check with %s: %s", t.agent.Name(), prompt))
		} else {
			verbose.Say(name, fmt.Sprintf("I need to check with %s: %s with images: %v", t.agent.Name(), prompt, imageIds))
		}
	}

	images := make([]string, 0, len(imageIds))
	for _, id := range imageIds {
		if img, ok := DefaultImageRegistry.GetImage(id); ok {
			images = append(images, img)
		} else {
			return nil, &scufris.Error{
				Code:    "IMAGE_NOT_FOUND",
				Message: fmt.Sprintf("image with id %s not found in registry", id),
				Err:     fmt.Errorf("image with id %s not found in registry", id),
			}
		}
	}

	return t.agent.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt).WithImages(images))
}
