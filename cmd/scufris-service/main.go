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
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/history"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/alexjercan/scufris/internal/protocol"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tool"
	"github.com/alexjercan/scufris/tools"
)

const SCUFRIS_AGENT_NAME = "Scufris"
const PLANNER_AGENT_NAME = "Planner"
const CODER_AGENT_NAME = "Coder"
const KNOWLEDGE_AGENT_NAME = "Knowledge"
const ARTIST_AGENT_NAME = "Artist"
const LLAVA_AGENT_NAME = "Llava"
const SHELL_AGENT_NAME = "Shell"

func main() {
	ctx := context.Background()
	protocol.MessageInit()

	config.SetupLogger(slog.LevelDebug, "text")
	logger := slog.Default()

	cfg, err := config.LoadConfig()
	if err != nil {
		panic(fmt.Errorf("failed to load config: %w", err))
	}

	client := llm.NewOllama(cfg.Ollama.Url)

	// Create the repositories
	db := config.GetDB(cfg)
	imageRepository := knowledge.NewImageRepository(db)
	embeddingRepository := knowledge.NewEmbeddingRepository(db)
	chunkRepository := knowledge.NewChunkRepository(db)
	knowledgeRepository := knowledge.NewKnowledgeRepository(db)
	sourceRepository := knowledge.NewKnowledgeSourceRepository(db)

	source, err := sourceRepository.GetByName(ctx, "transcript")
	if err != nil {
		fmt.Println("The transcript source does not exist.")
		return
	}

	// Create the knowledge worker - it will handle knowledge commands
	ch := make(chan knowledge.KnowledgeCommand, 100)
	worker := knowledge.NewKnowledgeWorker(ch)
	go worker.Start(ctx)

	// Factory to create knowledge commands
	commandFactory := knowledge.NewKnowledgeCommandFactory(
		chunkRepository,
		embeddingRepository,
		cfg.EmbeddingModel,
		client,
	)

	// Create the crew builder
	toolRegistry := tools.NewToolRegistry()
	imageRegistry := knowledge.NewKnowledgeImageRegistry(imageRepository)
	crew := agent.NewCrew(
		SCUFRIS_AGENT_NAME,
		client,
		toolRegistry,
		imageRegistry,
	)

	// Register the agents and their tools
	toolWeb, err := toolRegistry.RegisterTool(tools.NewWebSearchTool(5))
	if err != nil {
		panic(err)
	}
	toolWeather, err := toolRegistry.RegisterTool(tools.NewWeatherTool())
	if err != nil {
		panic(err)
	}
	toolOsList, err := toolRegistry.RegisterTool(tools.NewOsListTool())
	if err != nil {
		panic(err)
	}
	toolOsHome, err := toolRegistry.RegisterTool(tools.NewOsHomeTool())
	if err != nil {
		panic(err)
	}
	retriver := knowledge.NewRetriever(embeddingRepository, cfg.EmbeddingModel, client)
	retrieveTool, err := toolRegistry.RegisterTool(tools.NewRetrieveTool(5, retriver))
	if err != nil {
		panic(err)
	}
	imageGenerator := imagegen.NewSimple(cfg.ImageGen.Url)
	toolImageGen, err := toolRegistry.RegisterTool(tools.NewImageGeneratorTool(imageGenerator, imageRegistry))
	if err != nil {
		panic(err)
	}
	toolImageRead, err := toolRegistry.RegisterTool(tools.NewImageReadTool(imageRegistry))
	if err != nil {
		panic(err)
	}

	crew.RegisterAgent(
		SCUFRIS_AGENT_NAME,
		"The supervisor agent. The one and only Scufris the bestest LLM agent.",
		"scufris",
		[]string{},
		[]string{PLANNER_AGENT_NAME, CODER_AGENT_NAME, KNOWLEDGE_AGENT_NAME, ARTIST_AGENT_NAME, SHELL_AGENT_NAME},
	)

	crew.RegisterAgent(
		PLANNER_AGENT_NAME,
		"The planner agent. An expert at creating tasks and setting goals. This agent should be used to create a plan for complex tasks.",
		"planner",
		[]string{},
		[]string{},
	)

	crew.RegisterAgent(
		CODER_AGENT_NAME,
		"The coding agent. An expert at writing code.",
		"coder",
		[]string{},
		[]string{},
	)

	crew.RegisterAgent(
		KNOWLEDGE_AGENT_NAME,
		"The knowledge agent. An expert at searching for information. This agent can search the web. This agent can also retrieve information from the knowledge base, including past conversations.",
		"knowledge",
		[]string{toolWeb.Name, toolWeather.Name, retrieveTool.Name},
		[]string{},
	)

	crew.RegisterAgent(
		ARTIST_AGENT_NAME,
		"The artist agent. An expert at creating images and interpreting them. DO NO USE IMAGE_IDS",
		"artist",
		[]string{toolImageGen.Name, toolImageRead.Name},
		[]string{"Llava"},
	)

	crew.RegisterAgent(
		LLAVA_AGENT_NAME,
		"The vision agent. It analyzes images passed via `image_ids` and returns descriptions or analysis. Please use one image at a time.",
		"llava",
		[]string{},
		[]string{},
	)

	crew.RegisterAgent(
		SHELL_AGENT_NAME,
		"The shell agent. An expert at using the OS terminal. Can be interacted with using natural language.",
		"shell",
		[]string{toolOsList.Name, toolOsHome.Name},
		[]string{},
	)

	// Create the listener
	if err := os.RemoveAll(cfg.SocketPath); err != nil {
		panic(err)
	}
	listener, err := net.Listen("unix", cfg.SocketPath)
	if err != nil {
		panic(err)
	}
	logger.Info("Listen on: ", slog.String("SOCKET", cfg.SocketPath))

	// While we get connections, we will handle them
	for {
		fd, err := listener.Accept()
		if err != nil {
			logger.Error("Accept error: %w", slog.Any("Error", err))
			continue
		}

		go func(c net.Conn) {
			defer c.Close()

			ctx := context.Background()

			knowledgeId, err := knowledgeRepository.Create(ctx, knowledge.NewKnowledge(source.ID))
			if err != nil {
				logger.Error("Failed to create knowledge", slog.Any("Error", err))
				return
			}

			logger.Info("Handle new connection")

			enc := gob.NewEncoder(c)
			dec := gob.NewDecoder(c)

			// Create a new history callback handler
			transcript := &strings.Builder{}
			hc := history.NewHistoryCallback(transcript)
			defer func() {
				chunkID, err := chunkRepository.Create(ctx, knowledge.NewChunk(knowledgeId, 0))
				if err != nil {
					logger.Error("Failed to create chunk", slog.Any("Error", err))
					return
				}

				command := commandFactory.NewCreateCommand(chunkID, transcript.String())
				ch <- command

				logger.Debug("Transcript queued for processing", slog.String("Transcript", transcript.String()))
			}()

			// Create a new protocol callback handler
			pc := protocol.NewProtocolCallback(enc)

			// Setup the crew callbacks
			crew.OnStart = func(ctx context.Context, name string) error {
				if err := hc.OnStart(ctx, name); err != nil {
					return err
				}

				return pc.OnStart(ctx, name)
			}
			crew.OnToken = func(ctx context.Context, token string) error {
				if err := hc.OnToken(ctx, token); err != nil {
					return err
				}

				return pc.OnToken(ctx, token)
			}
			crew.OnEnd = func(ctx context.Context) error {
				if err := hc.OnEnd(ctx); err != nil {
					return err
				}

				return pc.OnEnd(ctx)
			}
			crew.OnImage = func(ctx context.Context, image string) error {
				return pc.OnImage(ctx, image)
			}
			crew.OnToolCall = func(ctx context.Context, name string, tool string, params tool.ToolParameters) error {
				if err := hc.OnToolCall(ctx, name, tool, params); err != nil {
					return err
				}

				return pc.OnToolCall(ctx, name, tool, params)
			}
			crew.OnToolResponse = func(ctx context.Context, name string, tool string, response tool.ToolResponse) error {
				if err := hc.OnToolResponse(ctx, name, tool, response); err != nil {
					return err
				}

				return pc.OnToolResponse(ctx, name, tool, response)
			}

			scufris, err := crew.Build(ctx)
			if err != nil {
				logger.Error("Failed to build crew", slog.Any("Error", err))
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
		}(fd)
	}
}
