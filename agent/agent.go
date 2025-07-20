package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

type Agent struct {
	name        string
	description string

	model    string
	llm      llm.Llm

	history []llm.Message
	tools   []llm.ToolInfo

	registry *tools.ToolRegistry
	config  *AgentConfig
}

type AgentConfig struct {
	IsVision bool `json:"is_vision,omitempty" jsonschema:"title=is_vision,description=Whether the agent supports vision or not."`
}

func NewAgentConfig() *AgentConfig {
	return &AgentConfig{
		IsVision: false, // Default to false, can be overridden
	}
}

func (c *AgentConfig) WithVision(isVision bool) *AgentConfig {
	c.IsVision = isVision
	return c
}

func NewAgent(name string, description string, model string, client llm.Llm, registry *tools.ToolRegistry, config *AgentConfig) *Agent {
	return &Agent{
		name:        name,
		description: description,

		model:    model,
		llm:      client,

		history: []llm.Message{},
		tools:   []llm.ToolInfo{},

		registry: registry,
		config:  config,
	}
}

func (a *Agent) Name() string {
	return a.name
}

func (a *Agent) Description() string {
	return a.description
}

func (a *Agent) IsVision() bool {
	if a.config == nil {
		return false
	}

	return a.config.IsVision
}

func (a *Agent) AddFunctionTool(tool tools.Tool) error {
	info, err := a.registry.RegisterTool(tool)
	if err != nil {
		name := tool.Name()
		return &scufris.Error{
			Code:    "TOOL_REGISTRATION_ERROR",
			Message: fmt.Sprintf("failed to register tool %s", name),
			Err:     fmt.Errorf("failed to register tool %s: %w", name, err),
		}
	}

	a.tools = append(a.tools, llm.ToolInfo{
		Type:     "function",
		Function: info,
	})

	return nil
}

func (a *Agent) AddMessage(message llm.Message) {
	a.history = append(a.history, message)
}

func (a *Agent) chat(ctx context.Context) (response string, err error) {
	ctx = contextkeys.WithAgentName(ctx, a.Name())

	result, err := a.llm.Chat(ctx, llm.NewChatRequest(a.model, a.history, a.tools, true))
	if err != nil {
		return response, err
	}

	m := result.Message
	a.AddMessage(m)

	if len(m.ToolCalls) > 0 {
		for _, toolCall := range m.ToolCalls {
			result, err := a.registry.CallTool(ctx, toolCall.Function.Name, toolCall.Function.Arguments)
			if err != nil {
				observer.OnError(ctx, err)

				promptStr := err.Error()
				a.AddMessage(llm.NewMessage(llm.RoleTool, promptStr))

				continue
			}

			prompt, err := json.Marshal(result)
			if err != nil {
				observer.OnError(ctx, err)

				promptStr := err.Error()
				a.AddMessage(llm.NewMessage(llm.RoleTool, promptStr))

				continue
			}

			promptStr := string(prompt)
			a.AddMessage(llm.NewMessage(llm.RoleTool, promptStr))
		}

		return a.chat(ctx)
	}

	return m.Content, nil
}

func (a *Agent) Chat(ctx context.Context, message llm.Message) (response string, err error) {
	a.AddMessage(message)

	return a.chat(ctx)
}
