package main

import (
	"context"
	"encoding/gob"
	"fmt"
	"log/slog"
	"net"
	"os"

	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/internal/socket"
	"github.com/alexjercan/scufris/llm"
)

func handleNewChat(c net.Conn) {
	logger := slog.Default()
	defer c.Close()

	logger.Info("Handle new connection")

	scufris := builder.Scufris()

	enc := gob.NewEncoder(c)
	dec := gob.NewDecoder(c)

	ctx := context.Background()
	ctx = observer.WithObserver(ctx, socket.NewSocketObserver(enc))
	ctx = registry.WithImageRegistry(ctx, registry.NewImageRegistry())

	// TODO: maybe defer to save the history of the chat on close
	// TODO: Create an observer that will create the transcript of the chat so that we can save it for later use
	// var database (or something to persists the chat history)
	// var transcript (something that implements io.Writer)
	// history.NewHistoryObserver(w *io.Writer) observer.Observer
	// defer database.SaveChatHistory(ctx, transcript)

	for {
		var m socket.Message
		if err := dec.Decode(&m); err != nil {
			logger.Error("Decode error: %w", slog.Any("Error", err))
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
				logger.Error("Encode error: %w", slog.Any("Error", err))
				return
			}
		default:
			logger.Error(fmt.Sprintf("unexpected socket.MessageKind: %#v", m.Kind))
			return
		}
	}
}

func main() {
	socket.MessageInit()

	logging.SetupLogger(slog.LevelDebug, "text")
	logger := slog.Default()

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

		go handleNewChat(fd)
	}
}
