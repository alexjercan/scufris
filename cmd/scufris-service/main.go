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

	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/callbacks"
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/alexjercan/scufris/internal/protocol"
	"github.com/alexjercan/scufris/llm"
)

func handle(ctx context.Context, cfg config.Config, c net.Conn) {
	defer c.Close()          // Ensure the connection is closed when done
	logger := slog.Default() // Use the default logger

	// Create encoders and decoders for the connection
	enc := gob.NewEncoder(c)
	dec := gob.NewDecoder(c)

	// Setup the database connection and repositories for this connection
	// and create the knowledge registry.
	// TODO: Might want to have a separate API for the knowledge registry
	db := config.GetDB(cfg)
	imageRepository := knowledge.NewImageRepository(db)
	embeddingRepository := knowledge.NewEmbeddingRepository(db)
	chunkRepository := knowledge.NewChunkRepository(db)
	knowledgeRepository := knowledge.NewKnowledgeRepository(db)
	sourceRepository := knowledge.NewKnowledgeSourceRepository(db)
	r := knowledge.NewKnowledgeRegistry(
		cfg.EmbeddingModel,
		llm.NewOllama(cfg.Ollama.Url),
		imageRepository,
		embeddingRepository,
		chunkRepository,
		knowledgeRepository,
		sourceRepository,
	)

	// Create a knowledge for this connection.
	transcript, err := sourceRepository.GetByName(ctx, "transcript") // TODO: Kind of hardcoded, but we can change this later
	if err != nil {
		logger.Error("Failed to get transcript knowledge source", slog.Any("Error", err))
		return
	}
	knowledgeID, err := knowledgeRepository.Create(ctx, knowledge.NewKnowledge(transcript.ID))
	if err != nil {
		logger.Error("Failed to create knowledge for transcript", slog.Any("Error", err))
		return
	}
	chunkID, err := chunkRepository.Create(ctx, knowledge.NewChunk(knowledgeID, 0))
	if err != nil {
		logger.Error("Failed to create chunk for transcript", slog.Any("Error", err))
		return
	}

	// Create a string builder that will be stored in the registry (in the transcript knowledge source)
	// This will be used to store the conversation transcript.
	t := &strings.Builder{}
	defer func() {
		if _, err := r.AddText(ctx, t.String(), &knowledge.TextOptions{ChunkID: chunkID}); err != nil {
			logger.Error("failed to add transcript to registry", slog.Any("error", err))
			return
		}
	}()

	hc := callbacks.NewHistoryCallback(t)    // Create a history callback to store the current conversation in a buffer
	pc := callbacks.NewProtocolCallback(enc) // Create a protocol callback to encode responses back to the client

	// Setup a new Scufris builder with the configuration and registry.
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

	// Very basic server loop to handle incoming messages from the client
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
			hc.OnPrompt(ctx, prompt) // We want to store the user prompt in the history callback

			response, err := scufris.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt))
			if err != nil {
				hc.OnError(ctx, err) // We want to store the error in the history callback
				if err := pc.OnError(ctx, err); err != nil {
					logger.Error("Encode error", slog.Any("Error", err))
					return
				}

				continue
			}

			// Send the response back to the client
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
	config.SetupLogger(slog.LevelDebug, "text") // Setup the logger
	protocol.MessageInit()                      // Initialize the protocol messages
	cfg := config.MustLoadConfig()              // Load the configuration
	logger := slog.Default()                    // Use the default logger

	// Create a Unix socket listener
	if err := os.RemoveAll(cfg.SocketPath); err != nil {
		panic(err)
	}
	listener, err := net.Listen("unix", cfg.SocketPath)
	if err != nil {
		panic(err)
	}
	logger.Info("Listen on: ", slog.String("SOCKET", cfg.SocketPath))

	// Very basic server loop
	for {
		ctx := context.Background() // Create a context for the connection handling

		fd, err := listener.Accept()
		if err != nil {
			logger.Error("Accept error: %w", slog.Any("Error", err))
			continue
		}

		go handle(ctx, cfg, fd)
	}
}
