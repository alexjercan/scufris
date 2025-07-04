package tools

import (
	"context"
	"log/slog"
)

type loggingTool struct {
	tool   Tool
	logger *slog.Logger
}

func (l *loggingTool) Name() string {
	return l.tool.Name()
}

func (l *loggingTool) Description() string {
	return l.tool.Description()
}

func (t *loggingTool) Call(ctx context.Context) (any, error) {
	t.logger.Debug("Tool.Call called",
		slog.String("name", t.Name()),
		slog.String("description", t.Description()),
	)

	result, err := t.tool.Call(ctx)
	if err != nil {
		t.logger.Error("Tool.Call error",
			slog.String("name", t.Name()),
			slog.String("error", err.Error()),
		)
		return nil, err
	}

	t.logger.Debug("Tool.Call completed",
		slog.String("name", t.Name()),
		slog.Any("result", result),
	)

	return result, nil
}
