package main

import (
	"context"
	"fmt"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tool"
	"github.com/alexjercan/scufris/tools"
)

type Observer struct{}

func main() {
	ctx := context.Background()
	client := llm.NewOllama("http://localhost:11434")
	registry := tools.NewToolRegistry()

	a := agent.NewAgent(
		"Example",
		"An example agent that uses the Ollama LLM.",
		"qwen3:latest",
		client,
		registry,
	)
	a.OnStart = func(ctx context.Context) error {
		fmt.Printf("%s: ", a.Name())
		return nil
	}
	a.OnToken = func(ctx context.Context, token string) error {
		fmt.Printf("%s", token)
		return nil
	}
	a.OnEnd = func(ctx context.Context) error {
		fmt.Println()
		return nil
	}
	a.OnToolCall = func(ctx context.Context, tool string, params tool.ToolParameters) error {
		s := params.String()
		if s == "" {
			fmt.Printf("%s: I will call the %s tool.\n", a.Name(), tool)
		} else {
			fmt.Printf("%s: I will call the %s tool with parameters: %s\n", a.Name(), tool, s)
		}

		return nil
	}
	a.OnToolResponse = func(ctx context.Context, tool string, response tool.ToolResponse) error {
		fmt.Printf("%s: The %s tool returned: %s\n", a.Name(), tool, response.String())
		return nil
	}

	info, err := registry.RegisterTool(tools.NewWeatherTool())
	if err != nil {
		panic(err)
	}
	a.AddFunctionTool(info)

	response, err := a.Chat(ctx, llm.NewMessage(llm.RoleUser, "What's the weather like in New York?"))
	if err != nil {
		panic(err)
	}

	fmt.Println("Response:", response)
}
