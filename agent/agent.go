package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

type Agent struct {
	name        string
	description string

	model    string
	llm      llm.Llm
	registry *tools.ToolRegistry

	history []llm.Message
	tools   []llm.ToolInfo
}

func NewAgent(name string, description string, model string, client llm.Llm, registry *tools.ToolRegistry) *Agent {
	return &Agent{
		name:        name,
		description: description,

		model:    model,
		llm:      client,
		registry: registry,

		history: []llm.Message{},
		tools:   []llm.ToolInfo{},
	}
}

func (a *Agent) Name() string {
	return a.name
}

func (a *Agent) Description() string {
	return a.description
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

	result, err := a.llm.Chat(ctx, llm.NewChatRequest(a.model, a.history, a.tools, true), nil)
	if err != nil {
		return
	}

	m := result.Message
	a.AddMessage(m)

	if len(m.ToolCalls) > 0 {
		var toolErrors []error

		for _, toolCall := range m.ToolCalls {
			result, err := a.registry.CallTool(ctx, toolCall.Function.Name, toolCall.Function.Arguments)
			if err != nil {
				toolErrors = append(toolErrors, err)
				continue
			}

			prompt, err := json.Marshal(result)
			if err != nil {
				toolErrors = append(toolErrors, &scufris.Error{
					Code:    "TOOL_CALL_MARSHAL_ERROR",
					Message: fmt.Sprintf("failed to marshal result for tool %s", toolCall.Function.Name),
					Err:     fmt.Errorf("failed to marshal result for tool %s: %w", toolCall.Function.Name, err),
				})
				continue
			}

			promptStr := string(prompt)
			a.AddMessage(llm.NewMessage(llm.RoleTool, promptStr))
		}

		if len(toolErrors) > 0 {
			return response, &scufris.Error{
				Code:    "TOOL_CALL_ERROR",
				Message: "one or more tool calls failed",
				Err:     fmt.Errorf("one or more tool calls failed: %v", toolErrors),
			}
		}

		response, err = a.chat(ctx)
		if err != nil {
			return response, &scufris.Error{
				Code:    "CHAT_ERROR",
				Message: "failed to continue chat after tool calls",
				Err:     fmt.Errorf("failed to continue chat after tool calls: %w", err),
			}
		}
	} else {
		response = m.Content
	}

	return response, err
}

func (a *Agent) Chat(ctx context.Context, prompt string) (response string, err error) {
	m := llm.NewMessage(llm.RoleUser, prompt)
	a.AddMessage(m)

	return a.chat(ctx)
}
