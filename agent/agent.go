package agent

import (
	"context"
	"fmt"
	"slices"

	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type Agent struct {
	name        string
	description string

	model string
	llm   llm.Llm

	history []llm.Message
	tools   []llm.ToolInfo

	registry tool.ToolRegistry

	// Callbacks for agent events
	OnStart func(context.Context) error
	OnToken func(context.Context, string) error
	OnEnd   func(context.Context) error

	OnImage        func(context.Context, uuid.UUID) error
	OnToolCall     func(context.Context, string, tool.ToolParameters) error
	OnToolResponse func(context.Context, string, tool.ToolResponse) error
}

func NewAgent(name string, description string, model string, client llm.Llm, registry tool.ToolRegistry) *Agent {
	return &Agent{
		name:        name,
		description: description,

		model: model,
		llm:   client,

		history: []llm.Message{},
		tools:   []llm.ToolInfo{},

		registry: registry,
	}
}

func (a *Agent) Name() string {
	return a.name
}

func (a *Agent) Description() string {
	return a.description
}

func (a *Agent) IsVision(ctx context.Context) bool {
	info, err := a.llm.ModelInfo(ctx, a.model)
	if err != nil {
		return false
	}

	return slices.Contains(info.Capabilities, llm.ModelInfoVision)
}

func (a *Agent) AddFunctionTool(info llm.FunctionToolInfo) {
	a.tools = append(a.tools, llm.ToolInfo{
		Type:     "function",
		Function: info,
	})
}

func (a *Agent) AddMessage(message llm.Message) {
	a.history = append(a.history, message)
}

func (a *Agent) chat(ctx context.Context) (response string, err error) {
	result, err := a.llm.Chat(ctx, llm.NewChatRequest(a.model, a.history, a.tools, true), &llm.ChatOptions{
		OnStart: a.OnStart,
		OnToken: a.OnToken,
		OnEnd:   a.OnEnd,
	})

	if err != nil {
		return response, err
	}

	m := result.Message
	a.AddMessage(m)

	if len(m.ToolCalls) > 0 {
		var toolErrors []error
		for _, toolCall := range m.ToolCalls {
			result, err := a.registry.CallTool(ctx, toolCall, &tool.ToolOptions{
				OnImage:        a.OnImage,
				OnToolCall:     a.OnToolCall,
				OnToolResponse: a.OnToolResponse,
			})
			if err != nil {
				toolErrors = append(toolErrors, err)

				promptStr := err.Error()
				a.AddMessage(llm.NewMessage(llm.RoleTool, promptStr))

				continue
			}

			prompt := result.String()
			a.AddMessage(llm.NewMessage(llm.RoleTool, prompt))
		}

		if len(toolErrors) > 0 {
			return response, &Error{
				Code:    "TOOL_CALL_ERROR",
				Message: "one or more tool calls failed",
				Err:     fmt.Errorf("one or more tool calls failed: %v", toolErrors),
			}
		}

		return a.chat(ctx)
	}

	return m.Content, nil
}

func (a *Agent) Chat(ctx context.Context, message llm.Message) (response string, err error) {
	a.AddMessage(message)

	return a.chat(ctx)
}
