package main

import (
	"context"
	"encoding/gob"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"

	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/history"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/internal/socket"
	"github.com/alexjercan/scufris/llm"
)

func main() {
	ctx := context.Background()
	socket.MessageInit()

	logging.SetupLogger(slog.LevelDebug, "text")
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
	tw := history.NewDbTranscriptWriter(db, ch)

	go worker.Start(ctx)

	if err := os.Remove(cfg.SocketPath); err != nil {
		panic(err)
	}

	listener, err := net.Listen("unix", cfg.SocketPath)
	if err != nil {
		panic(err)
	}

	logger.Info("Listen on: ", slog.String("SOCKET", cfg.SocketPath))

	for {
		fd, err := listener.Accept()
		if err != nil {
			logger.Error("Accept error: %w", slog.Any("Error", err))
			continue
		}

		go func(c net.Conn) {
			logger := slog.Default()
			defer c.Close()

			ctx := context.Background()
			logger.Info("Handle new connection")

			scufris := builder.Scufris(client, imageGenerator, retriever)

			enc := gob.NewEncoder(c)
			dec := gob.NewDecoder(c)

			defer func() {
				if err := tw.Close(); err != nil {
					logger.Error("Failed to close transcript writer", slog.Any("Error", err))
				}
			}()

			ctx = observer.WithObserver(ctx, socket.NewSocketObserver(enc), history.NewHistoryObserver(tw))
			ctx = registry.WithImageRegistry(ctx, registry.NewDbImageRegistry(db))

			for {
				var m socket.Message
				if err := dec.Decode(&m); err != nil {
					if err == io.EOF {
						logger.Info("Client closed the connection")
						return
					}

					logger.Error("Decode error", slog.Any("Error", err))
					return
				}

				switch m.Kind {
				case socket.MessagePrompt:
					prompt := m.Payload.(socket.PayloadPrompt).Prompt
					observer.OnUser(ctx, prompt)
					response, err := scufris.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt))
					if err != nil {
						observer.OnError(ctx, err)
					}

					err = enc.Encode(socket.NewMessage(socket.MessageResponse, socket.PayloadResponse{Response: response}))
					if err != nil {
						logger.Error("Encode error", slog.Any("Error", err))
						return
					}
				default:
					logger.Error(fmt.Sprintf("Unexpected socket.MessageKind: %#v", m.Kind))
					return
				}
			}
		}(fd)
	}
}
