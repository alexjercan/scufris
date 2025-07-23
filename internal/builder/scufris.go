package builder

import (
	"context"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/tools"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/registry"
	"github.com/alexjercan/scufris/tool"
)

const SCUFRIS_AGENT_NAME = "Scufris"
const PLANNER_AGENT_NAME = "Planner"
const CODER_AGENT_NAME = "Coder"
const KNOWLEDGE_AGENT_NAME = "Knowledge"
const ARTIST_AGENT_NAME = "Artist"
const LLAVA_AGENT_NAME = "Llava"
const SHELL_AGENT_NAME = "Shell"

type ScufrisBuilder struct {
	client    llm.Llm
	tools     tool.ToolRegistry
	registry  registry.Registry
	imagegen  imagegen.ImageGenerator
	callbacks []agent.CrewCallbacks
}

func NewScufrisBuilder(cfg config.Config) *ScufrisBuilder {
	return &ScufrisBuilder{
		client:    llm.NewOllama(cfg.Ollama.Url),
		tools:     tool.NewMapToolRegistry(),
		registry:  registry.NewMapRegistry(nil, nil),
		imagegen:  imagegen.NewSimple(cfg.ImageGen.Url),
		callbacks: []agent.CrewCallbacks{},
	}
}

func (b *ScufrisBuilder) WithClient(client llm.Llm) *ScufrisBuilder {
	b.client = client
	return b
}

func (b *ScufrisBuilder) WithToolRegistry(tools tool.ToolRegistry) *ScufrisBuilder {
	b.tools = tools
	return b
}

func (b *ScufrisBuilder) WithRegistry(registry registry.Registry) *ScufrisBuilder {
	b.registry = registry
	return b
}

func (b *ScufrisBuilder) WithImageGenerator(imagegen imagegen.ImageGenerator) *ScufrisBuilder {
	b.imagegen = imagegen
	return b
}

func (b *ScufrisBuilder) WithCallbacks(callbacks ...agent.CrewCallbacks) *ScufrisBuilder {
	b.callbacks = append(b.callbacks, callbacks...)
	return b
}

func (b *ScufrisBuilder) Build(ctx context.Context) (*agent.Agent, error) {
	crew := agent.NewCrew(
		SCUFRIS_AGENT_NAME,
		b.client,
		b.tools,
		b.registry,
	)

	toolWeb, err := b.tools.RegisterTool(tools.NewWebSearchTool(5))
	if err != nil {
		panic(err)
	}
	toolWeather, err := b.tools.RegisterTool(tools.NewWeatherTool())
	if err != nil {
		panic(err)
	}
	toolOsList, err := b.tools.RegisterTool(tools.NewOsListTool())
	if err != nil {
		panic(err)
	}
	toolOsHome, err := b.tools.RegisterTool(tools.NewOsHomeTool())
	if err != nil {
		panic(err)
	}
	toolRetriever, err := b.tools.RegisterTool(tools.NewRetrieveTool(5, b.registry))
	if err != nil {
		panic(err)
	}
	toolImageGen, err := b.tools.RegisterTool(tools.NewImageGeneratorTool(b.imagegen, b.registry))
	if err != nil {
		panic(err)
	}
	toolImageRead, err := b.tools.RegisterTool(tools.NewImageReadTool(b.registry))
	if err != nil {
		panic(err)
	}

	crew.RegisterAgent(
		SCUFRIS_AGENT_NAME,
		"The supervisor agent. The one and only Scufris the bestest LLM agent.",
		"scufris",
		[]string{},
		[]string{PLANNER_AGENT_NAME, CODER_AGENT_NAME, KNOWLEDGE_AGENT_NAME, ARTIST_AGENT_NAME, SHELL_AGENT_NAME},
	)

	crew.RegisterAgent(
		PLANNER_AGENT_NAME,
		"The planner agent. An expert at creating tasks and setting goals. This agent should be used to create a plan for complex tasks.",
		"planner",
		[]string{},
		[]string{},
	)

	crew.RegisterAgent(
		CODER_AGENT_NAME,
		"The coding agent. An expert at writing code.",
		"coder",
		[]string{},
		[]string{},
	)

	crew.RegisterAgent(
		KNOWLEDGE_AGENT_NAME,
		"The knowledge agent. An expert at searching for information. This agent can search the web. This agent can also retrieve information from the knowledge base, including past conversations.",
		"knowledge",
		[]string{toolWeb.Name, toolWeather.Name, toolRetriever.Name},
		[]string{},
	)

	crew.RegisterAgent(
		ARTIST_AGENT_NAME,
		"The artist agent. An expert at creating images and interpreting them. DO NO USE IMAGE_IDS",
		"artist",
		[]string{toolImageGen.Name, toolImageRead.Name},
		[]string{"Llava"},
	)

	crew.RegisterAgent(
		LLAVA_AGENT_NAME,
		"The vision agent. It analyzes images passed via `image_ids` and returns descriptions or analysis. Please use one image at a time.",
		"llava",
		[]string{},
		[]string{},
	)

	crew.RegisterAgent(
		SHELL_AGENT_NAME,
		"The shell agent. An expert at using the OS terminal. Can be interacted with using natural language.",
		"shell",
		[]string{toolOsList.Name, toolOsHome.Name},
		[]string{},
	)

	crew.OnStart = func(ctx context.Context, agentName string) error {
		for _, cb := range b.callbacks {
			if cb.OnStart != nil {
				if err := cb.OnStart(ctx, agentName); err != nil {
					return err
				}
			}
		}
		return nil
	}
	crew.OnToken = func(ctx context.Context, token string) error {
		for _, cb := range b.callbacks {
			if cb.OnToken != nil {
				if err := cb.OnToken(ctx, token); err != nil {
					return err
				}
			}
		}
		return nil
	}
	crew.OnEnd = func(ctx context.Context) error {
		for _, cb := range b.callbacks {
			if cb.OnEnd != nil {
				if err := cb.OnEnd(ctx); err != nil {
					return err
				}
			}
		}
		return nil
	}
	crew.OnImage = func(ctx context.Context, image string) error {
		for _, cb := range b.callbacks {
			if cb.OnImage != nil {
				if err := cb.OnImage(ctx, image); err != nil {
					return err
				}
			}
		}
		return nil
	}
	crew.OnToolCall = func(ctx context.Context, agentName string, toolName string, params tool.ToolParameters) error {
		for _, cb := range b.callbacks {
			if cb.OnToolCall != nil {
				if err := cb.OnToolCall(ctx, agentName, toolName, params); err != nil {
					return err
				}
			}
		}
		return nil
	}
	crew.OnToolResponse = func(ctx context.Context, agentName string, toolName string, response tool.ToolResponse) error {
		for _, cb := range b.callbacks {
			if cb.OnToolResponse != nil {
				if err := cb.OnToolResponse(ctx, agentName, toolName, response); err != nil {
					return err
				}
			}
		}
		return nil
	}

	return crew.Build(ctx)
}
