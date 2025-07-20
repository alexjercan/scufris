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
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/internal/socket"
	"github.com/alexjercan/scufris/llm"
	"github.com/uptrace/bun"
)

func handleNewChat(c net.Conn, client llm.Llm, db *bun.DB, ch chan<- knowledge.KnowledgeChanItem) {
	logger := slog.Default()
	defer c.Close()

	ctx := context.Background()
	logger.Info("Handle new connection")

	scufris := builder.Scufris(client)
	enc := gob.NewEncoder(c)
	dec := gob.NewDecoder(c)

	tw := history.NewDbTranscriptWriter(db, ch)
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
}

func main() {
	ctx := context.Background()
	socket.MessageInit()

	logging.SetupLogger(slog.LevelDebug, "text")
	logger := slog.Default()

	db := config.GetDB()
	client := llm.NewOllama(config.OLLAMA_URL)
	ch := make(chan knowledge.KnowledgeChanItem, 100)
	worker := knowledge.NewKnowledgeWorker(db, ch, config.EMBEDDING_MODEL, client)

	go worker.Start(ctx)

	if err := os.Remove(socket.SOCKET_PATH); err != nil {
		panic(err)
	}

	listener, err := net.Listen("unix", socket.SOCKET_PATH)
	if err != nil {
		panic(err)
	}

	logger.Info("Listen on: ", slog.String("SOCKET", socket.SOCKET_PATH))

	for {
		fd, err := listener.Accept()
		if err != nil {
			logger.Error("Accept error: %w", slog.Any("Error", err))
			continue
		}

		go handleNewChat(fd, client, db, ch)
	}
}
