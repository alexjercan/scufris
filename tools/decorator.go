package tools

import "log/slog"

type ToolWrapper struct {
	tool Tool
}

func NewToolWrapper(tool Tool) *ToolWrapper {
	return &ToolWrapper{
		tool: tool,
	}
}

func (w *ToolWrapper) WithLogging(logger *slog.Logger) *ToolWrapper {
	w.tool = &loggingTool{
		tool:   w.tool,
		logger: logger,
	}
	return w
}

func (w *ToolWrapper) Build() Tool {
	return w.tool
}
