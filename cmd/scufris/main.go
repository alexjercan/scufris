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

	scufris := agent.NewAgent("Scufris", "The supervisor agent. The one and only Scufris the bestest LLM agent.", "scufris", client, registry)
	coder := agent.NewAgent("Coder", "A coding expert agent.", "deepseek-r1", client, registry)

	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewWeatherTool()).WithLogging(slog.Default()).Build())
	scufris.AddFunctionTool(tools.NewToolWrapper(tools.NewDelegateTool(coder)).WithLogging(slog.Default()).Build())

	scufris.AddMessage(llm.NewMessage(llm.RoleSystem, "You are a helpful assistant. You can answer questions and use tools to provide information."))
	coder.AddMessage(llm.NewMessage(llm.RoleSystem, "You are a helpful coding assistant. You must provide code answers."))

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
