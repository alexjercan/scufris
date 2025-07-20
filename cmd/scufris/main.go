package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/history"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/internal/verbose"
	"github.com/alexjercan/scufris/llm"
)

func main() {
	ctx := context.Background()

	logging.SetupLogger(slog.LevelInfo, "text")
	logger := slog.Default()

	cfg, err := config.LoadConfig()
	if err != nil {
		panic(fmt.Errorf("failed to load config: %w", err))
	}

	ch := make(chan knowledge.KnowledgeChanItem, 100)
	db := config.GetDB(cfg)
	client := llm.NewOllama(cfg.Ollama.Url)
	worker := knowledge.NewKnowledgeWorker(db, ch, cfg.EmbeddingModel, client)
	imageGenerator := imagegen.NewSimple(cfg.ImageGen.Url)
	retriever := knowledge.NewRetriever(db, cfg.EmbeddingModel, client)

	go worker.Start(ctx)

	scufris := builder.Scufris(client, imageGenerator, retriever)

	tw := history.NewDbTranscriptWriter(db, ch)
	defer func() {
		if err := tw.Close(); err != nil {
			logger.Error("Failed to close transcript writer", slog.Any("Error", err))
		}
	}()

	ctx = observer.WithObserver(ctx, verbose.NewVerboseObserver(), history.NewHistoryObserver(tw))
	ctx = registry.WithImageRegistry(ctx, registry.NewDbImageRegistry(db))

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
			return
		}
		observer.OnUser(ctx, prompt)

		response, err := scufris.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt))
		if err != nil {
			observer.OnError(ctx, err)
		}

		fmt.Println(response)
	}
}
