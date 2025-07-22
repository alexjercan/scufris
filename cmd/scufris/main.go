package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/history"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/pretty"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tool"
	"github.com/alexjercan/scufris/tools"
)

const SCUFRIS_AGENT_NAME = "Scufris"
const PLANNER_AGENT_NAME = "Planner"
const CODER_AGENT_NAME = "Coder"
const KNOWLEDGE_AGENT_NAME = "Knowledge"
const ARTIST_AGENT_NAME = "Artist"
const LLAVA_AGENT_NAME = "Llava"
const SHELL_AGENT_NAME = "Shell"

func main() {
	ctx := context.Background()

	config.SetupLogger(slog.LevelInfo, "text")
	logger := slog.Default()

	cfg, err := config.LoadConfig()
	if err != nil {
		panic(fmt.Errorf("failed to load config: %w", err))
	}

	client := llm.NewOllama(cfg.Ollama.Url)

	// Create the crew builder
	toolRegistry := tools.NewToolRegistry()
	imageRegistry := registry.NewMapImageRegistry()
	crew := agent.NewCrew(
		SCUFRIS_AGENT_NAME,
		client,
		toolRegistry,
		imageRegistry,
	)

	// Register the agents and their tools
	toolWeb, err := toolRegistry.RegisterTool(tools.NewWebSearchTool(5))
	if err != nil {
		panic(err)
	}
	toolWeather, err := toolRegistry.RegisterTool(tools.NewWeatherTool())
	if err != nil {
		panic(err)
	}
	toolOsList, err := toolRegistry.RegisterTool(tools.NewOsListTool())
	if err != nil {
		panic(err)
	}
	toolOsHome, err := toolRegistry.RegisterTool(tools.NewOsHomeTool())
	if err != nil {
		panic(err)
	}
	imageGenerator := imagegen.NewSimple(cfg.ImageGen.Url)
	toolImageGen, err := toolRegistry.RegisterTool(tools.NewImageGeneratorTool(imageGenerator, imageRegistry))
	if err != nil {
		panic(err)
	}
	toolImageRead, err := toolRegistry.RegisterTool(tools.NewImageReadTool(imageRegistry))
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
		[]string{toolWeb.Name, toolWeather.Name},
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

	// Create a new history callback handler
	transcript := &strings.Builder{}
	hc := history.NewHistoryCallback(transcript)
	defer func() {
		if err := os.WriteFile("transcript.txt", []byte(transcript.String()), 0644); err != nil {
			logger.Error("failed to write transcript to file", slog.Any("error", err))
			return
		}

		logger.Debug("Transcript saved",
			slog.String("path", "transcript.txt"),
		)
	}()

	crew.OnStart = func(ctx context.Context, name string) error {
		if err := hc.OnStart(ctx, name); err != nil {
			return err
		}

		pretty.PrintName(name)
		fmt.Printf(": ")

		return nil
	}
	crew.OnToken = func(ctx context.Context, token string) error {
		if err := hc.OnToken(ctx, token); err != nil {
			return err
		}

		pretty.PrintToken(token)

		return nil
	}
	crew.OnEnd = func(ctx context.Context) error {
		if err := hc.OnEnd(ctx); err != nil {
			return err
		}

		fmt.Println()

		return nil
	}
	crew.OnImage = func(ctx context.Context, image string) error {
		pretty.PrintImage(image)
		fmt.Println()

		return nil
	}
	crew.OnToolCall = func(ctx context.Context, name string, tool string, params tool.ToolParameters) error {
		if err := hc.OnToolCall(ctx, name, tool, params); err != nil {
			return err
		}

		s := params.String()
		if s == "" {
			pretty.PrintName(name)
			fmt.Printf(": ")
			pretty.PrintToken(fmt.Sprintf("I will call the %s tool.", tool))
			fmt.Println()
		} else {
			pretty.PrintName(name)
			fmt.Printf(": ")
			pretty.PrintToken(fmt.Sprintf("I will call the %s tool with parameters: %s", tool, s))
			fmt.Println()
		}

		return nil
	}

	crew.OnToolResponse = func(ctx context.Context, name string, tool string, response tool.ToolResponse) error {
		if err := hc.OnToolResponse(ctx, name, tool, response); err != nil {
			return err
		}

		pretty.PrintName(name)
		fmt.Printf(": ")
		pretty.PrintToken(fmt.Sprintf("The %s tool returned: %s", tool, response.String()))
		fmt.Println()

		return nil
	}

	scufris, err := crew.Build(ctx)
	if err != nil {
		logger.Error("failed to build Scufris agent", slog.Any("error", err))
		return
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		hc.OnPrompt(ctx, prompt)
		if err != nil {
			fmt.Println(err)
			return
		}

		response, err := scufris.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt))
		if err != nil {
			hc.OnError(ctx, err)
			fmt.Printf("Error: %v\n", err)
		}

		fmt.Println(response)
	}
}

// TODO: Add a webscraping tool
// TODO: Add references in the text provided by knowledge agent
// TODO: Add agent for interpreting data from somewhere
// TODO: Add PDF Parsing Tool
// TODO: Implement the rest of the functions for CRUD chunks
// TODO: Come up with a way to have scufris somehow start in scratchpad
// TODO: Some kind of CLI utlity that I can use to add knowledge sources/delete things,
// basically a CRUD thingy tool for utils... web based IDK, for now, just CLI
// TODO: Add status on top of chunk so that we can set it to failed or something like that
// we can try to add new tables for storing errors `ChunkError` `KnowledgeError`
// TODO: Maybe small cleanup for the code IDK...
// TODO: Handle all errors in `handleCreate` - set status to failed or add some kind of error message in the chunk/knowledge
// TODO: Add image in knowledge package
// TODO: Remove context stuff...
// TODO: Move all config to config
