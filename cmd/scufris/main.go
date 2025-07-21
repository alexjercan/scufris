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
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/internal/verbose"
	"github.com/alexjercan/scufris/llm"
)

func main() {
	ctx := context.Background()

	config.SetupLogger(slog.LevelInfo, "text")
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
