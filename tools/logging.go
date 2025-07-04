package tools

import (
	"context"
	"log/slog"
	"reflect"
)

type loggingTool struct {
	tool   Tool
	logger *slog.Logger
}

func (l *loggingTool) Parameters() reflect.Type {
	return l.tool.Parameters()
}

func (l *loggingTool) Name() string {
	return l.tool.Name()
}

func (l *loggingTool) Description() string {
	return l.tool.Description()
}

func (t *loggingTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	t.logger.Debug("Tool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	result, err := t.tool.Call(ctx, params)
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
