package config

import (
	"fmt"
	"log/slog"
	"os"
)

func SetupLogger(level slog.Level, format string) error {
	var handler slog.Handler

	// Choose the appropriate handler
	switch format {
	case "json":
		handler = slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: level,
		})
	case "text":
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: level,
		})
	default:
		return fmt.Errorf("unsupported log format: %s", format)
	}

	logger := slog.New(handler)
	slog.SetDefault(logger)

	return nil
}
