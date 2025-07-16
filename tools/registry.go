package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/llm"
)

type ToolFactory func(ctx context.Context, arguments map[string]any) (any, error)

type ToolRegistry struct {
	registry map[string]ToolFactory

	logger *slog.Logger
}

func NewToolRegistry(logger *slog.Logger) *ToolRegistry {
	if logger == nil {
		logger = slog.Default()
	}

	return &ToolRegistry{
		registry: make(map[string]ToolFactory),
		logger:   logger,
	}
}

func (r *ToolRegistry) CallTool(ctx context.Context, name string, arguments map[string]any) (any, error) {
	r.logger.Debug("ToolRegistry.CallTool called",
		slog.String("name", name),
		slog.Any("arguments", arguments),
	)

	call, ok := r.registry[name]
	if !ok {
		return nil, &scufris.Error{
			Code:    "TOOL_NOT_FOUND",
			Message: fmt.Sprintf("tool %s not found", name),
			Err:     fmt.Errorf("tool %s not found in registry", name),
		}
	}

	return call(ctx, arguments)
}

func (r *ToolRegistry) RegisterTool(tool Tool) (llm.FunctionToolInfo, error) {
	name := tool.Name()
	description := tool.Description()
	paramsType := tool.Parameters()

	r.logger.Debug("ToolRegistry.RegisterTool called",
		slog.String("name", name),
		slog.String("description", description),
		slog.Any("paramsType", paramsType),
	)

	paramPtr := reflect.New(paramsType)
	if _, ok := paramPtr.Interface().(ToolParameters); !ok {
		return llm.FunctionToolInfo{}, &scufris.Error{
			Code:    "INVALID_TOOL_PARAMETERS",
			Message: fmt.Sprintf("tool %s parameters must implement ToolParameters interface", tool.Name()),
			Err:     fmt.Errorf("tool %s parameters must implement ToolParameters interface", tool.Name()),
		}
	}

	if _, ok := r.registry[name]; !ok {
		r.registry[name] = func(ctx context.Context, arguments map[string]any) (any, error) {
			raw, err := json.Marshal(arguments)
			if err != nil {
				return nil, &scufris.Error{
					Code:    "TOOL_ARGUMENTS_MARSHAL_ERROR",
					Message: fmt.Sprintf("failed to marshal arguments for tool %s", name),
					Err:     fmt.Errorf("failed to marshal arguments for tool %s: %w", name, err),
				}
			}

			if err := json.Unmarshal(raw, paramPtr.Interface()); err != nil {
				return nil, &scufris.Error{
					Code:    "TOOL_ARGUMENTS_UNMARSHAL_ERROR",
					Message: fmt.Sprintf("failed to unmarshal arguments for tool %s", name),
					Err:     fmt.Errorf("failed to unmarshal arguments for tool %s: %w", name, err),
				}
			}

			v := paramPtr.Interface().(ToolParameters)
			if err := v.Validate(); err != nil {
				return nil, &scufris.Error{
					Code:    "TOOL_ARGUMENTS_VALIDATION_ERROR",
					Message: fmt.Sprintf("invalid arguments for tool %s: %v", name, err),
					Err:     fmt.Errorf("invalid arguments for tool %s: %w", name, err),
				}
			}

			return tool.Call(ctx, v)
		}
	}

	r.logger.Debug("ToolRegistry.RegisterTool completed")

	return llm.NewFunctionToolInfo(name, description, paramPtr.Interface()), nil
}

type ImageRegistry struct {
	registry map[string]string
	logger   *slog.Logger
}

func NewImageRegistry(logger *slog.Logger) *ImageRegistry {
	if logger == nil {
		logger = slog.Default()
	}

	return &ImageRegistry{
		registry: make(map[string]string),
		logger:   logger,
	}
}

func (r *ImageRegistry) AddImage(data string) string {
	imageId := fmt.Sprintf("image_%d", len(r.registry))
	r.registry[imageId] = data

	r.logger.Debug("ImageRegistry.AddImage called",
		slog.String("imageId", imageId),
		slog.Int("totalImages", len(r.registry)),
	)

	return imageId
}

func (r *ImageRegistry) GetImage(id string) (string, bool) {
	r.logger.Debug("ImageRegistry.GetImage called",
		slog.String("imageId", id),
	)

	img, ok := r.registry[id]
	if !ok {
		r.logger.Warn("ImageRegistry.GetImage not found",
			slog.String("imageId", id),
		)
		return "", false
	}

	return img, true
}

var DefaultImageRegistry = NewImageRegistry(nil)
