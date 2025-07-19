package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/alexjercan/scufris/internal/builder"
	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/internal/verbose"
	"github.com/alexjercan/scufris/llm"
)

func main() {
	logging.SetupLogger(slog.LevelInfo, "text")

	scufris := builder.Scufris()

	ctx := context.Background()
	ctx = observer.WithObserver(ctx, verbose.NewVerboseObserver())
	ctx = registry.WithImageRegistry(ctx, registry.NewImageRegistry())

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
			return
		}
		response, err := scufris.Chat(ctx, llm.NewMessage(llm.RoleUser, prompt))
		if err != nil {
			observer.OnError(ctx, err)
		}

		fmt.Println(response)
	}
}
