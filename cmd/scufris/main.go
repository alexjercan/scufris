package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

const OLLAMA_URL = "http://localhost:11434"

func main() {
	logging.SetupLogger(slog.LevelInfo, "text")

	ctx := context.Background()

	registry := tools.NewToolRegistry(nil)
	client := llm.NewLlmWrapper(llm.NewOllama(OLLAMA_URL)).WithLogging(slog.Default()).Build()

	scufris := agent.NewAgent(
		"Scufris",
		"The supervisor agent. The one and only Scufris the bestest LLM agent.",
		"scufris",
		client,
		registry,
	)
	planner := agent.NewAgent(
		"Planner",
		"The planner agent. An expert at creating tasks and setting goals.",
		"planner",
		client,
		registry,
	)
	coder := agent.NewAgent(
		"Coder",
		"The coding agent. An expert at writing code.",
		"coder",
		client,
		registry,
	)
	knowledge := agent.NewAgent(
		"Knowledge",
		"The knowledge agent. An expert at searching for information.",
		"knowledge",
		client,
		registry,
	)

	// TODO: have an agent for tools like weather
	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewWeatherTool()).WithLogging(slog.Default()).Build())

	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(planner)).WithLogging(slog.Default()).Build())
	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(coder)).WithLogging(slog.Default()).Build())
	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(knowledge)).WithLogging(slog.Default()).Build())

	knowledge.AddFunctionTool(tools.NewToolWrapper(tools.NewWebSearchTool(5)).WithLogging(slog.Default()).Build())
	// TODO: Add a webscraping tool
	// TODO: Add references in the text provided by knowledge agent

	// TODO: Add agent for interpreting data from somewhere
	// TODO: Add PDF Parsing Tool

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
			return
		}
		response, err := scufris.Chat(ctx, prompt)
		if err != nil {
			fmt.Println(err)
			return
		}

		fmt.Println(response)
	}
}
