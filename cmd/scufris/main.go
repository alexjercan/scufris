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
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/verbose"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

const OLLAMA_URL = "http://localhost:11434"
const IMAGEGEN_URL = "http://localhost:8080"

func main() {
	logging.SetupLogger(slog.LevelInfo, "text")

	ctx := context.Background()
	ctx = observer.WithObserver(ctx, verbose.NewVerboseObserver())

	client := llm.NewOllama(OLLAMA_URL)
	imageGenerator := imagegen.NewSimple(IMAGEGEN_URL)

	scufris := agent.NewAgent(
		"Scufris",
		"The supervisor agent. The one and only Scufris the bestest LLM agent.",
		"scufris",
		client,
	)
	planner := agent.NewAgent(
		"Planner",
		"The planner agent. An expert at creating tasks and setting goals. This agent should be used to create a plan for complex tasks.",
		"planner",
		client,
	)
	coder := agent.NewAgent(
		"Coder",
		"The coding agent. An expert at writing code.",
		"coder",
		client,
	)
	knowledge := agent.NewAgent(
		"Knowledge",
		"The knowledge agent. An expert at searching for information. This agent can search the web.",
		"knowledge",
		client,
	)
	artist := agent.NewAgent(
		"Artist",
		"The artist agent. An expert at handling images from paths or IDs. It can also generate images based on prompts that it can improve. It can also load images from path.",
		"artist",
		client,
	)
	llava := agent.NewAgent(
		"Llava",
		"The image interpreter agent. VISION AGENT ONLY!",
		"llava",
		client,
	)

	// TODO: have an agent for tools like weather same for image generation
	scufris.AddFunctionTool(ctx, tools.NewWeatherTool())

	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(planner))
	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(coder))
	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(knowledge))
	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(artist))

	knowledge.AddFunctionTool(ctx, tools.NewWebSearchTool(5))
	// TODO: Add a webscraping tool
	// TODO: Add references in the text provided by knowledge agent

	// TODO: Add agent for interpreting data from somewhere
	// TODO: Add PDF Parsing Tool

	artist.AddFunctionTool(ctx, tools.NewImageGeneratorTool(imageGenerator))
	artist.AddFunctionTool(ctx, tools.NewDelegateTool(llava))
	artist.AddFunctionTool(ctx, tools.NewImageReadTool())

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
