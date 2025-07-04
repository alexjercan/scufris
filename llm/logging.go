package llm

import (
	"context"
	"log/slog"
)

type loggingLlm struct {
	llm    Llm
	logger *slog.Logger
}

func (l *loggingLlm) Chat(ctx context.Context, request ChatRequest) (ChatResponse, error) {
	l.logger.Debug("Ollama.Chat called",
		slog.String("model", request.Model),
		slog.Int("messages", len(request.Messages)),
		slog.Int("tools", len(request.Tools)),
		slog.Bool("stream", request.Stream),
	)

	resp, err := l.llm.Chat(ctx, request)
	if err != nil {
		l.logger.Error("Ollama.Chat", slog.String("error", err.Error()))
		return resp, err
	}

	l.logger.Debug("Ollama.Chat response",
		slog.String("content", resp.Message.Content),
		slog.Any("tool_calls", resp.Message.ToolCalls),
	)

	return resp, nil
}
