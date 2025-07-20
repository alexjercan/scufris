package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/internal/verbose"
	"github.com/alexjercan/scufris/llm"
)

func main() {
	logging.SetupLogger(slog.LevelInfo, "text")

	cfg, err := config.LoadConfig()
	if err != nil {
		panic(fmt.Errorf("failed to load config: %w", err))
	}

	client := llm.NewOllama(cfg.Ollama.Url)
	imageGenerator := imagegen.NewSimple(cfg.ImageGen.Url)

	scufris := builder.Scufris(client, imageGenerator)

	ctx := context.Background()
	ctx = observer.WithObserver(ctx, verbose.NewVerboseObserver())
	ctx = registry.WithImageRegistry(ctx, registry.NewMapImageRegistry())

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
			observer.OnError(ctx, err)
		}

		fmt.Println(response)
	}
}
