package main

import (
	"context"
	"fmt"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tool"
	"github.com/alexjercan/scufris/tools"
)

type Observer struct{}

func main() {
	ctx := context.Background()
	client := llm.NewOllama("http://localhost:11434")

	toolRegistry := tools.NewToolRegistry()
	imageRegistry := registry.NewMapImageRegistry()

	crew := agent.NewCrew(
		"Example",
		client,
		toolRegistry,
		imageRegistry,
	)

	crew.RegisterAgent(
		"Example",
		"An example agent that uses the Ollama LLM.",
		"qwen3:latest",
		[]string{},
		[]string{"ExpertChef"},
	)

	crew.RegisterAgent(
		"ExpertChef",
		"A culinary expert agent that can provide cooking advice and recipes.",
		"qwen3:latest",
		[]string{},
		[]string{},
	)

	crew.OnStart = func(ctx context.Context, name string) error {
		fmt.Printf("%s: ", name)
		return nil
	}
	crew.OnToken = func(ctx context.Context, token string) error {
		fmt.Printf("%s", token)
		return nil
	}
	crew.OnEnd = func(ctx context.Context) error {
		fmt.Println()
		return nil
	}
	crew.OnToolCall = func(ctx context.Context, name string, tool string, params tool.ToolParameters) error {
		s := params.String()
		if s == "" {
			fmt.Printf("%s: I will call the %s tool.\n", name, tool)
		} else {
			fmt.Printf("%s: I will call the %s tool with parameters: %s\n", name, tool, s)
		}

		return nil
	}
	crew.OnToolResponse = func(ctx context.Context, name string, tool string, response tool.ToolResponse) error {
		fmt.Printf("%s: The %s tool returned: %s\n", name, tool, response.String())
		return nil
	}

	a, err := crew.Build(ctx)
	if err != nil {
		panic(err)
	}

	response, err := a.Chat(ctx, llm.NewMessage(llm.RoleUser, "Give me a recipe for a delicious pasta dish."))
	if err != nil {
		panic(err)
	}

	fmt.Println("Response:", response)
}
