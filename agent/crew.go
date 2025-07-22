package agent

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/registry"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type AgentInfo struct {
	Name        string
	Description string
	Model       string

	Tools     []string
	Delegates []string
}

type CrewCallbacks struct {
	OnStart func(context.Context, string) error
	OnToken func(context.Context, string) error
	OnEnd   func(context.Context) error

	OnImage        func(context.Context, string) error
	OnToolCall     func(context.Context, string, string, tool.ToolParameters) error
	OnToolResponse func(context.Context, string, string, tool.ToolResponse) error
}

type Crew struct {
	supervisor string
	agents     map[string]AgentInfo
	client     llm.Llm
	tools      tool.ToolRegistry
	registry   registry.Registry

	logger *slog.Logger

	CrewCallbacks
}

func NewCrew(
	supervisor string,
	client llm.Llm,
	tools tool.ToolRegistry,
	registry registry.Registry,
) *Crew {
	return &Crew{
		supervisor: supervisor,
		agents:     make(map[string]AgentInfo),
		client:     client,
		tools:      tools,
		registry:   registry,

		logger: slog.Default(),
	}
}

func (c *Crew) RegisterAgent(
	name string,
	description string,
	model string,
	tools []string,
	delegates []string,
) {
	c.logger.Debug("Registering agent",
		slog.String("name", name),
		slog.String("description", description),
		slog.String("model", model),
		slog.Any("tools", tools),
		slog.Any("delegates", delegates),
	)

	c.agents[name] = AgentInfo{
		Name:        name,
		Description: description,
		Model:       model,
		Tools:       tools,
		Delegates:   delegates,
	}
}

func (c *Crew) Build(ctx context.Context) (*Agent, error) {
	agents := make(map[string]*Agent, len(c.agents))

	for name, info := range c.agents {
		agent := NewAgent(
			info.Name,
			info.Description,
			info.Model,
			c.client,
			c.tools,
		)

		agent.OnStart = func(ctx context.Context) error {
			if c.OnStart != nil {
				return c.OnStart(ctx, name)
			}
			return nil
		}
		agent.OnToken = c.OnToken
		agent.OnEnd = c.OnEnd
		agent.OnImage = func(ctx context.Context, imageID uuid.UUID) error {
			if c.OnImage != nil {
				image, err := c.registry.GetImage(ctx, imageID)
				if err != nil {
					return &scufris.Error{
						Code:    "IMAGE_NOT_FOUND",
						Message: fmt.Sprintf("image %s not found", imageID),
						Err:     fmt.Errorf("image %s not found in registry: %w", imageID, err),
					}
				}

				return c.OnImage(ctx, image)
			}

			return nil
		}
		agent.OnToolCall = func(ctx context.Context, s string, tp tool.ToolParameters) error {
			if c.OnToolCall != nil {
				return c.OnToolCall(ctx, name, s, tp)
			}
			return nil
		}
		agent.OnToolResponse = func(ctx context.Context, s string, tr tool.ToolResponse) error {
			if c.OnToolResponse != nil {
				return c.OnToolResponse(ctx, name, s, tr)
			}
			return nil
		}

		agents[name] = agent

		for _, tool := range info.Tools {
			info, err := c.tools.FunctionToolInfo(tool)
			if err != nil {
				return nil, err
			}

			agents[name].AddFunctionTool(info)
		}
	}

	for name, info := range c.agents {
		for _, delegate := range info.Delegates {
			if agent, ok := agents[delegate]; ok {
				info, err := c.tools.RegisterTool(
					NewDelegateTool(agent, c.registry),
				)
				if err != nil {
					return nil, err
				}
				agents[name].AddFunctionTool(info)
			}
		}
	}

	return agents[c.supervisor], nil
}
