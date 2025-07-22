package agent

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/registry"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type AgentLike interface {
	Name() string
	Description() string
	IsVision(ctx context.Context) bool

	Chat(ctx context.Context, message llm.Message) (string, error)
}

type DelegateToolParameters struct {
	Prompt   string   `json:"prompt" jsonschema:"title=prompt,description=The prompt to use for the delegate agent."`
	ImageIds []string `json:"image_ids,omitempty" jsonschema:"title=image_ids,description=The image ids to use with the delegate agent. THIS SHOULD BE USED BY VISION AGENTS ONLY!"`
}

func (p *DelegateToolParameters) Validate(tool tool.Tool) error {
	if p.Prompt == "" {
		return fmt.Errorf("prompt cannot be empty")
	}

	return nil
}

func (p *DelegateToolParameters) String() string {
	if len(p.ImageIds) > 0 {
		return fmt.Sprintf("prompt: %s, image_ids: %v", p.Prompt, p.ImageIds)
	}
	return fmt.Sprintf("prompt: %s", p.Prompt)
}

type DelegateToolResponse struct {
	Result string `json:"result" jsonschema:"title=result,description=The result of the delegate agent's response."`
}

func (r *DelegateToolResponse) String() string {
	return fmt.Sprintf("%s", r.Result)
}

func (r *DelegateToolResponse) Image() uuid.UUID {
	return uuid.Nil
}

type DelegateTool struct {
	agent    AgentLike
	registry registry.Registry
	logger   *slog.Logger
}

func NewDelegateTool(agent AgentLike, registry registry.Registry) tool.Tool {
	return &DelegateTool{
		agent:    agent,
		registry: registry,
		logger:   slog.Default(),
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

func (t *DelegateTool) Call(ctx context.Context, params tool.ToolParameters) (tool.ToolResponse, error) {
	t.logger.Debug("DelegateTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	prompt := params.(*DelegateToolParameters).Prompt
	imageIds := params.(*DelegateToolParameters).ImageIds

	if !t.agent.IsVision(ctx) && len(imageIds) > 0 {
		prompt = fmt.Sprintf("%s with images: %v", prompt, imageIds)
		imageIds = []string{} // Clear imageIds if agent is not vision
	}

	images := make([]string, 0, len(imageIds))
	for _, id := range imageIds {
		imageId := uuid.MustParse(id)
		img, err := t.registry.GetImage(ctx, imageId)
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

	return &DelegateToolResponse{
		Result: result,
	}, nil
}
