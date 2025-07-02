package scufris

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

type Agent struct {
	model string
	llm   llm.Llm

	history []llm.Message
	tools   []llm.ToolInfo

	logger *slog.Logger
}

func NewAgent(model string, client llm.Llm) *Agent {
	return &Agent{
		model: model,
		llm:   client,

		history: []llm.Message{},
		tools:   []llm.ToolInfo{},
		logger:  slog.Default(),
	}
}

func (a *Agent) AddFunctionTool(tool tools.Tool) error {
	a.logger.Debug("Agent.AddFunctionTool called", "name", tool.Name(), "description", tool.Description())

	info, err := tools.RegisterTool(tool)
	if err != nil {
		name := tool.Name()
		return fmt.Errorf("failed to register tool %s: %w", name, err)
	}

	a.tools = append(a.tools, llm.ToolInfo{
		Type:     "function",
		Function: info,
	})

	a.logger.Debug("Agent.AddFunctionTool completed")

	return nil
}

func (a *Agent) AddMessage(message llm.Message) {
	a.logger.Debug("Agent.AddMessage called", "role", message.Role, "content", message.Content)

	a.history = append(a.history, message)
}

func (a *Agent) chat(ctx context.Context, role llm.MessageRole, prompt string) (response string, err error) {
	m := llm.NewMessage(role, prompt)
	a.logger.Debug("Agent.chat called", "role", m.Role, "content", m.Content)
	a.AddMessage(m)

	result, err := a.llm.Chat(ctx, llm.NewChatRequest(a.model, a.history, a.tools, false))
	if err != nil {
		return
	}

	m = result.Message
	a.AddMessage(m)

	if len(m.ToolCalls) > 0 {
		a.logger.Debug("Agent.chat processing tool calls", "count", len(m.ToolCalls))

		for _, toolCall := range m.ToolCalls {
			a.logger.Debug("Agent.chat processing tool call", "name", toolCall.Function.Name, "arguments", toolCall.Function.Arguments)

			function, err := tools.GetTool(toolCall.Function.Name, toolCall.Function.Arguments)
			if err != nil {
				a.logger.Error("Error getting function tool", "name", toolCall.Function.Name, "error", err)
				continue
			}

			result, err := function.Call(ctx)
			if err != nil {
				a.logger.Error("Error calling function tool", "name", toolCall.Function.Name, "error", err)
				continue
			}

			prompt, err := json.Marshal(result)
			if err != nil {
				a.logger.Error("Error marshaling function tool result", "name", toolCall.Function.Name, "error", err)
				continue
			}

			promptStr := string(prompt)
			response, err = a.chat(ctx, llm.RoleTool, promptStr)
		}
	} else {
		response = m.Content
	}

	a.logger.Debug("Agent.chat completed", "response", response)

	return response, err
}

func (a *Agent) Chat(ctx context.Context, prompt string) (response string, err error) {
	return a.chat(ctx, llm.RoleUser, prompt)
}
