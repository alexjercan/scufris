package builder

import (
	"context"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

const OLLAMA_URL = "http://localhost:11434"
const IMAGEGEN_URL = "http://localhost:8080"

func Scufris(ctx context.Context) *agent.Agent {
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
		"The artist agent. Does not take image_ids. Only works with textual prompts or local image paths.",
		"artist",
		client,
	)
	llava := agent.NewAgent(
		"Llava",
		"The vision agent. It analyzes images passed via `image_ids` and returns descriptions or analysis.",
		"llava",
		client,
	)
	shell := agent.NewAgent(
		"Shell",
		"The shell agent. An expert at using the OS terminal. Can be interacted with using natural language.",
		"shell",
		client,
	)

	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(planner))
	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(coder))
	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(knowledge))
	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(artist))
	scufris.AddFunctionTool(ctx, tools.NewDelegateTool(shell))

	knowledge.AddFunctionTool(ctx, tools.NewWebSearchTool(5))
	knowledge.AddFunctionTool(ctx, tools.NewWeatherTool())
	// TODO: Add a webscraping tool
	// TODO: Add references in the text provided by knowledge agent
	// TODO: Add agent for interpreting data from somewhere
	// TODO: Add PDF Parsing Tool

	artist.AddFunctionTool(ctx, tools.NewImageGeneratorTool(imageGenerator))
	artist.AddFunctionTool(ctx, tools.NewDelegateTool(llava))
	artist.AddFunctionTool(ctx, tools.NewImageReadTool())

	shell.AddFunctionTool(ctx, tools.NewOsListTool())
	shell.AddFunctionTool(ctx, tools.NewHomeTool())

	return scufris
}
