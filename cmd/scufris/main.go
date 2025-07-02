package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tools"
)

const OLLAMA_URL = "http://localhost:11434"
const MODEL_NAME = "llama3.2:1b"

func main() {
	logging.SetupLogger(slog.LevelDebug, "text")

	ctx := context.Background()

	client := llm.NewOllama(OLLAMA_URL)
	agent := scufris.NewAgent(MODEL_NAME, client)

	agent.AddFunctionTool(tools.NewWeatherTool())

	agent.AddMessage(llm.NewMessage(llm.RoleSystem, "You are a helpful assistant. You can answer questions and use tools to provide information."))

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
			return
		}
		response, err := agent.Chat(ctx, prompt)
		if err != nil {
			fmt.Println(err)
			return
		}

		fmt.Println(response)
	}
}
