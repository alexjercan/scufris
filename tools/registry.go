package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type ToolCall func(ctx context.Context, arguments map[string]any, opts *tool.ToolOptions) (tool.ToolResponse, error)

type RegistryEntry struct {
	Call             ToolCall
	FunctionToolInfo llm.FunctionToolInfo
}

type ToolRegistry struct {
	registry map[string]RegistryEntry

	logger *slog.Logger
}

func NewToolRegistry() tool.ToolRegistry {
	return &ToolRegistry{
		registry: make(map[string]RegistryEntry),
		logger:   slog.Default(),
	}
}

func (r *ToolRegistry) CallTool(ctx context.Context, call llm.ToolCall, opts *tool.ToolOptions) (tool.ToolResponse, error) {
	name := call.Function.Name
	arguments := call.Function.Arguments

	r.logger.Debug("ToolRegistry.CallTool called",
		slog.String("name", name),
		slog.Any("arguments", arguments),
	)

	ent, ok := r.registry[name]
	if !ok {
		return nil, &scufris.Error{
			Code:    "TOOL_NOT_FOUND",
			Message: fmt.Sprintf("tool %s not found", name),
			Err:     fmt.Errorf("tool %s not found in registry", name),
		}
	}

	return ent.Call(ctx, arguments, opts)
}

func (r *ToolRegistry) FunctionToolInfo(name string) (llm.FunctionToolInfo, error) {
	r.logger.Debug("ToolRegistry.FunctionToolInfo called",
		slog.String("name", name),
	)

	ent, ok := r.registry[name]
	if !ok {
		return llm.FunctionToolInfo{}, &scufris.Error{
			Code:    "TOOL_NOT_FOUND",
			Message: fmt.Sprintf("tool %s not found", name),
			Err:     fmt.Errorf("tool %s not found in registry", name),
		}
	}

	r.logger.Debug("ToolRegistry.FunctionToolInfo completed",
		slog.String("name", name),
		slog.Any("info", ent.FunctionToolInfo),
	)

	return ent.FunctionToolInfo, nil
}

func (r *ToolRegistry) RegisterTool(t tool.Tool) (llm.FunctionToolInfo, error) {
	name := t.Name()
	description := t.Description()
	paramsType := t.Parameters()

	r.logger.Debug("ToolRegistry.RegisterTool called",
		slog.String("name", name),
		slog.String("description", description),
		slog.Any("paramsType", paramsType),
	)

	paramPtr := reflect.New(paramsType)
	if _, ok := paramPtr.Interface().(tool.ToolParameters); !ok {
		return llm.FunctionToolInfo{}, &scufris.Error{
			Code:    "INVALID_TOOL_PARAMETERS",
			Message: fmt.Sprintf("tool %s parameters must implement ToolParameters interface", t.Name()),
			Err:     fmt.Errorf("tool %s parameters must implement ToolParameters interface", t.Name()),
		}
	}

	info := llm.NewFunctionToolInfo(name, description, paramPtr.Interface())

	if _, ok := r.registry[name]; !ok {
		call := func(ctx context.Context, arguments map[string]any, opts *tool.ToolOptions) (tool.ToolResponse, error) {
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

			v := paramPtr.Interface().(tool.ToolParameters)
			if err := v.Validate(t); err != nil {
				return nil, &scufris.Error{
					Code:    "TOOL_ARGUMENTS_VALIDATION_ERROR",
					Message: fmt.Sprintf("invalid arguments for tool %s: %v", name, err),
					Err:     fmt.Errorf("invalid arguments for tool %s: %w", name, err),
				}
			}

			if opts != nil && opts.OnToolCall != nil {
				if err := opts.OnToolCall(ctx, name, v); err != nil {
					return nil, err
				}
			}

			result, err := t.Call(ctx, v)
			if err != nil {
				return nil, err
			}

			if imageID := result.Image(); imageID != uuid.Nil && opts != nil && opts.OnImage != nil {
				if err := opts.OnImage(ctx, imageID); err != nil {
					return nil, err
				}
			}

			if opts != nil && opts.OnToolResponse != nil {
				if err := opts.OnToolResponse(ctx, name, result); err != nil {
					return nil, err
				}
			}

			r.logger.Debug("ToolRegistry.CallTool completed",
				slog.String("name", name),
				slog.Any("result", result),
			)

			return result, nil
		}

		r.registry[name] = RegistryEntry{
			Call:             call,
			FunctionToolInfo: info,
		}
	}

	r.logger.Debug("ToolRegistry.RegisterTool completed")

	return info, nil
}
