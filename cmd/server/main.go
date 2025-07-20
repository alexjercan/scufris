package main

import (
	"context"
	"database/sql"
	"encoding/gob"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"

	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/history"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/internal/socket"
	"github.com/alexjercan/scufris/llm"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/sqlitedialect"
	"github.com/uptrace/bun/driver/sqliteshim"
)

func handleNewChat(c net.Conn, db *bun.DB) {
	logger := slog.Default()
	defer c.Close()

	logger.Info("Handle new connection")

	scufris := builder.Scufris()
	enc := gob.NewEncoder(c)
	dec := gob.NewDecoder(c)

	// TODO: Implement a transcript writer that saves the conversation to a RAG database
	// Then we can have a tool that can fetch information from the old conversations
	tw := history.NewFileTranscriptWriter("transcript.txt")
	defer tw.Close()

	ctx := context.Background()
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
	socket.MessageInit()

	logging.SetupLogger(slog.LevelDebug, "text")
	logger := slog.Default()

	sqldb, err := sql.Open(sqliteshim.ShimName, "file:test.sqlite?cache=shared")
	if err != nil {
		panic(err)
	}

	db := bun.NewDB(sqldb, sqlitedialect.New())

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

		go handleNewChat(fd, db)
	}
}
