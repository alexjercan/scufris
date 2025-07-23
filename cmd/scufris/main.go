package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/callbacks"
	"github.com/alexjercan/scufris/internal/pretty"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/registry"
	"github.com/alexjercan/scufris/tool"
)

func main() {
	ctx := context.Background()

	config.SetupLogger(slog.LevelInfo, "text")
	logger := slog.Default()

	cfg, err := config.LoadConfig()
	if err != nil {
		panic(fmt.Errorf("failed to load config: %w", err))
	}

	// Variable options
	r := registry.NewMapRegistry(nil, &registry.MapTextOptions{Path: "transcript.txt"})
	t := &strings.Builder{}
	defer func() {
		if _, err := r.AddText(ctx, t.String(), r.GetTextOptions()); err != nil {
			logger.Error("failed to add transcript to registry", slog.Any("error", err))
			return
		}
	}()

	hc := callbacks.NewHistoryCallback(t)

	b := builder.NewScufrisBuilder(cfg)
	b.WithRegistry(r)

	b.WithCallbacks(
		hc.ToCallbacks(),
		agent.CrewCallbacks{
			OnStart: func(ctx context.Context, s string) error {
				pretty.PrintName(s)
				fmt.Printf(": ")
				return nil
			},
			OnToken: func(ctx context.Context, token string) error {
				pretty.PrintToken(token)
				return nil
			},
			OnEnd: func(ctx context.Context) error {
				fmt.Println()
				return nil
			},
			OnImage: func(ctx context.Context, image string) error {
				pretty.PrintImage(image)
				fmt.Println()
				return nil
			},
			OnToolCall: func(ctx context.Context, name string, tool string, params tool.ToolParameters) error {
				s := params.String()
				if s == "" {
					pretty.PrintName(name)
					fmt.Printf(": ")
					pretty.PrintToken(fmt.Sprintf("I will call the %s tool.", tool))
					fmt.Println()
				} else {
					pretty.PrintName(name)
					fmt.Printf(": ")
					pretty.PrintToken(fmt.Sprintf("I will call the %s tool with parameters: %s", tool, s))
					fmt.Println()
				}
				return nil
			},
			OnToolResponse: func(ctx context.Context, name string, tool string, response tool.ToolResponse) error {
				pretty.PrintName(name)
				fmt.Printf(": ")
				pretty.PrintToken(fmt.Sprintf("The %s tool returned: %s", tool, response.String()))
				fmt.Println()
				return nil
			},
		},
	)

	scufris, err := b.Build(ctx)
	if err != nil {
		panic(err)
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
			return
		}
		hc.OnPrompt(ctx, prompt)

		response, err := scufris.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt))
		if err != nil {
			hc.OnError(ctx, err)
			fmt.Printf("Error: %v\n", err)
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
