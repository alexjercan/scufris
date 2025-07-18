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
	"github.com/alexjercan/scufris/internal/socket"
	"github.com/alexjercan/scufris/llm"
)

func handleNewChat(c net.Conn) {
	logger := slog.Default()
	defer c.Close()

	logger.Info("Handle new connection")

	enc := gob.NewEncoder(c)
	dec := gob.NewDecoder(c)

	ctx := context.Background()
	obs := socket.NewSocketObserver(enc)
	ctx = observer.WithObserver(ctx, obs)

	scufris := builder.Scufris(ctx)

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
				obs.OnError(ctx, err)
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
