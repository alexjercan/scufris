package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

const OLLAMA_URL = "http://localhost:11434"
const IMAGEGEN_URL = "http://localhost:8080"

func main() {
	logging.SetupLogger(slog.LevelInfo, "text")

	ctx := context.Background()

	registry := tools.NewToolRegistry(nil)
	client := llm.NewLlmWrapper(llm.NewOllama(OLLAMA_URL)).WithLogging(slog.Default()).WithVerbose().Build()
	imageGenerator := imagegen.NewImageGeneratorWrapper(imagegen.NewSimple(IMAGEGEN_URL)).WithLogging(slog.Default()).Build()

	scufris := agent.NewAgent(
		"Scufris",
		"The supervisor agent. The one and only Scufris the bestest LLM agent.",
		"scufris",
		client,
		registry,
	)
	planner := agent.NewAgent(
		"Planner",
		"The planner agent. An expert at creating tasks and setting goals. This agent should be used to create a plan for complex tasks.",
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
		"The knowledge agent. An expert at searching for information. This agent can search the web.",
		"knowledge",
		client,
		registry,
	)
	artist := agent.NewAgent(
		"Artist",
		"The artist agent. An expert at handling images from paths or IDs. It can also generate images based on prompts that it can improve. It can also load images from path.",
		"artist",
		client,
		registry,
	)
	llava := agent.NewAgent(
		"Llava",
		"The image interpreter agent. VISION AGENT ONLY!",
		"llava",
		client,
		registry,
	)

	// TODO: have an agent for tools like weather same for image generation
	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewWeatherTool()).WithLogging(slog.Default()).Build())

	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(planner)).WithLogging(slog.Default()).Build())
	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(coder)).WithLogging(slog.Default()).Build())
	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(knowledge)).WithLogging(slog.Default()).Build())
	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(artist)).WithLogging(slog.Default()).Build())

	knowledge.AddFunctionTool(tools.NewToolWrapper(tools.NewWebSearchTool(5)).WithLogging(slog.Default()).Build())
	// TODO: Add a webscraping tool
	// TODO: Add references in the text provided by knowledge agent

	// TODO: Add agent for interpreting data from somewhere
	// TODO: Add PDF Parsing Tool

	artist.AddFunctionTool(tools.NewToolWrapper(tools.NewImageTool(imageGenerator)).WithLogging(slog.Default()).Build())
	artist.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(llava)).WithLogging(slog.Default()).Build())
	artist.AddFunctionTool(tools.NewToolWrapper(tools.NewReadImageTool()).WithLogging(slog.Default()).Build())

	// shell.AddFunctionTool(tools.NewToolWrapper(tools.NewReadTool()).WithLogging(slog.Default()).Build())
	// shell.AddFunctionTool(tools.NewToolWrapper(tools.NewWriteTool()).WithLogging(slog.Default()).Build())

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
			return
		}
		response, err := scufris.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt))
		if err != nil {
			fmt.Println(err)
			return
		}

		fmt.Println(response)
	}
}
