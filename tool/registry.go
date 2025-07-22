package tool

import (
	"context"

	"github.com/alexjercan/scufris/llm"
	"github.com/google/uuid"
)

type ToolRegistry interface {
	RegisterTool(t Tool) (llm.FunctionToolInfo, error)
	CallTool(ctx context.Context, call llm.ToolCall, opts *ToolOptions) (ToolResponse, error)

	FunctionToolInfo(name string) (llm.FunctionToolInfo, error)
}

type ToolOptions struct {
	OnImage        func(ctx context.Context, imageID uuid.UUID) error
	OnToolCall     func(ctx context.Context, name string, params ToolParameters) error
	OnToolResponse func(ctx context.Context, name string, response ToolResponse) error
}
