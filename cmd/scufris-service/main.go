package main

import (
	"context"
	"encoding/gob"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"strings"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/callbacks"
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/alexjercan/scufris/internal/protocol"
	"github.com/alexjercan/scufris/llm"
)

func handle(cfg config.Config, c net.Conn) {
	defer c.Close()
	ctx := context.Background()
	logger := slog.Default()

	enc := gob.NewEncoder(c)
	dec := gob.NewDecoder(c)

	db := config.GetDB(cfg)
	imageRepository := knowledge.NewImageRepository(db)
	embeddingRepository := knowledge.NewEmbeddingRepository(db)
	chunkRepository := knowledge.NewChunkRepository(db)
	knowledgeRepository := knowledge.NewKnowledgeRepository(db)
	sourceRepository := knowledge.NewKnowledgeSourceRepository(db)

	r := knowledge.NewKnowledgeRegistry(
		imageRepository,
		embeddingRepository,
		chunkRepository,
		knowledgeRepository,
		sourceRepository,
	)
	t := &strings.Builder{}
	defer func() {
		if _, err := r.AddText(ctx, t.String(), &knowledge.TextOptions{Source: "transcript"}); err != nil {
			logger.Error("failed to add transcript to registry", slog.Any("error", err))
			return
		}
	}()

	hc := callbacks.NewHistoryCallback(t)
	pc := callbacks.NewProtocolCallback(enc)

	b := builder.NewScufrisBuilder(cfg)
	b.WithRegistry(r)

	b.WithCallbacks(
		hc.ToCallbacks(),
		pc.ToCallbacks(),
	)

	scufris, err := b.Build(ctx)
	if err != nil {
		logger.Error("Failed to build Scufris", slog.Any("Error", err))
		return
	}

	for {
		var m protocol.Message
		if err := dec.Decode(&m); err != nil {
			if err == io.EOF {
				logger.Info("Client closed the connection")
				return
			}

			logger.Error("Decode error", slog.Any("Error", err))
			return
		}

		switch m.Kind {
		case protocol.MessagePrompt:
			prompt := m.Payload.(protocol.PayloadPrompt).Prompt
			hc.OnPrompt(ctx, prompt)

			response, err := scufris.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt))
			if err != nil {
				hc.OnError(ctx, err)
				if err := pc.OnError(ctx, err); err != nil {
					logger.Error("Encode error", slog.Any("Error", err))
					return
				}

				continue
			}

			if err := pc.OnResponse(ctx, response); err != nil {
				logger.Error("Encode error", slog.Any("Error", err))
				return
			}
		default:
			logger.Error(fmt.Sprintf("Unexpected protocol.MessageKind: %#v", m.Kind))
			return
		}
	}
}

func main() {
	protocol.MessageInit()

	config.SetupLogger(slog.LevelDebug, "text")
	logger := slog.Default()

	cfg, err := config.LoadConfig()
	if err != nil {
		panic(fmt.Errorf("failed to load config: %w", err))
	}

	if err := os.RemoveAll(cfg.SocketPath); err != nil {
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

		go handle(cfg, fd)
	}
}
