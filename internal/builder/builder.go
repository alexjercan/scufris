package builder

import (
	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

func Scufris(client llm.Llm, gen imagegen.ImageGenerator, retriever *knowledge.Retriever) *agent.Agent {
	toolRegistry := tools.NewToolRegistry()

	scufris := agent.NewAgent(
		"Scufris",
		"The supervisor agent. The one and only Scufris the bestest LLM agent.",
		"scufris",
		client,
		toolRegistry,
		agent.NewAgentConfig(),
	)
	planner := agent.NewAgent(
		"Planner",
		"The planner agent. An expert at creating tasks and setting goals. This agent should be used to create a plan for complex tasks.",
		"planner",
		client,
		toolRegistry,
		agent.NewAgentConfig(),
	)
	coder := agent.NewAgent(
		"Coder",
		"The coding agent. An expert at writing code.",
		"coder",
		client,
		toolRegistry,
		agent.NewAgentConfig(),
	)
	knowledge := agent.NewAgent(
		"Knowledge",
		"The knowledge agent. An expert at searching for information. This agent can search the web. This agent can also retrieve information from the knowledge base, including past conversations.",
		"knowledge",
		client,
		toolRegistry,
		agent.NewAgentConfig(),
	)
	artist := agent.NewAgent(
		"Artist",
		"The artist agent. An expert at creating images and interpreting them. DO NO USE IMAGE_IDS",
		"artist",
		client,
		toolRegistry,
		agent.NewAgentConfig(),
	)
	llava := agent.NewAgent(
		"Llava",
		"The vision agent. It analyzes images passed via `image_ids` and returns descriptions or analysis. Please use one image at a time.",
		"llava",
		client,
		toolRegistry,
		agent.NewAgentConfig().WithVision(true),
	)
	shell := agent.NewAgent(
		"Shell",
		"The shell agent. An expert at using the OS terminal. Can be interacted with using natural language.",
		"shell",
		client,
		toolRegistry,
		agent.NewAgentConfig(),
	)

	scufris.AddFunctionTool(tools.NewDelegateTool(planner))
	scufris.AddFunctionTool(tools.NewDelegateTool(coder))
	scufris.AddFunctionTool(tools.NewDelegateTool(knowledge))
	scufris.AddFunctionTool(tools.NewDelegateTool(artist))
	scufris.AddFunctionTool(tools.NewDelegateTool(shell))

	knowledge.AddFunctionTool(tools.NewWebSearchTool(5))
	knowledge.AddFunctionTool(tools.NewWeatherTool())
	knowledge.AddFunctionTool(tools.NewRetrieveTool(5, retriever))
	// TODO: Add a webscraping tool
	// TODO: Add references in the text provided by knowledge agent
	// TODO: Add agent for interpreting data from somewhere
	// TODO: Add PDF Parsing Tool

	artist.AddFunctionTool(tools.NewImageGeneratorTool(gen))
	artist.AddFunctionTool(tools.NewDelegateTool(llava))
	artist.AddFunctionTool(tools.NewImageReadTool())

	shell.AddFunctionTool(tools.NewOsListTool())
	shell.AddFunctionTool(tools.NewHomeTool())

	return scufris
}
