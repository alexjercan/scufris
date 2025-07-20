package tools

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/llm"
)

type AgentLike interface {
	Name() string
	Description() string
	IsVision() bool

	Chat(ctx context.Context, message llm.Message) (string, error)
}

type DelegateToolParameters struct {
	Prompt   string   `json:"prompt" jsonschema:"title=prompt,description=The prompt to use for the delegate agent."`
	ImageIds []string `json:"image_ids,omitempty" jsonschema:"title=image_ids,description=The image ids to use with the delegate agent. THIS SHOULD BE USED BY VISION AGENTS ONLY!"`
}

func (p *DelegateToolParameters) Validate(tool Tool) error {
	if p.Prompt == "" {
		return fmt.Errorf("prompt cannot be empty")
	}

	if len(p.ImageIds) > 0 {
		if !tool.(*DelegateTool).agent.IsVision() {
			return fmt.Errorf("this is not a vision agent, image_ids should not be used")
		}
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
	t.logger.Debug("DelegateTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	prompt := params.(*DelegateToolParameters).Prompt
	imageIds := params.(*DelegateToolParameters).ImageIds

	observer.OnStart(ctx)
	text := fmt.Sprintf("I need to check with %s: %s", t.agent.Name(), prompt)
	if len(imageIds) > 0 {
		text = fmt.Sprintf("I need to check with %s: %s with images: %v", t.agent.Name(), prompt, imageIds)
	}
	err := observer.OnToken(ctx, text)
	if err != nil {
		return nil, err
	}
	observer.OnEnd(ctx)

	images := make([]string, 0, len(imageIds))
	for _, id := range imageIds {
		img, err := registry.GetImage(ctx, id)
		if err != nil {
			return nil, err
		}

		images = append(images, img)
	}

	result, err := t.agent.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt).WithImages(images))
	if err != nil {
		return nil, err
	}

	t.logger.Debug("DelegateTool.Call completed",
		slog.String("name", t.Name()),
		slog.Any("result", result),
	)

	return result, nil
}
