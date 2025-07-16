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
	"github.com/alexjercan/scufris/internal/verbose"
	"github.com/alexjercan/scufris/llm"
)

func main() {
	logging.SetupLogger(slog.LevelInfo, "text")

	ctx := context.Background()
	obs := verbose.NewVerboseObserver()
	ctx = observer.WithObserver(ctx, obs)

	scufris := builder.Scufris(ctx)

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
			obs.OnError(ctx, err)
		}

		fmt.Println(response)
	}
}
