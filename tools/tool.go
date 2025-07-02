package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris/llm"
)

type Tool interface {
	Name() string
	Description() string

	Call(ctx context.Context) (any, error)
}

type ToolParameters interface {
	Validate() error
}

type ToolFactory func(map[string]any) (Tool, error)

var logger = slog.Default()
var registry = map[string]ToolFactory{}

func GetTool(name string, arguments map[string]any) (Tool, error) {
	factory, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("tool %s not registered", name)
	}
	return factory(arguments)
}

// TODO: Maybe make this function into a method for some struct instead of a global function?
func RegisterTool(tool Tool) (llm.FunctionToolInfo, error) {
	name := tool.Name()
	description := tool.Description()

	logger.Debug("RegisterTool called", "name", name, "description", description)

	val := reflect.ValueOf(tool).Elem()
	field := val.FieldByName("Params")
	if !field.IsValid() {
		toolType := reflect.TypeOf(tool)
		return llm.FunctionToolInfo{}, fmt.Errorf("tool type %s has no Params field", toolType.Name())
	}

	paramPtr := reflect.New(field.Type())

	if _, ok := registry[name]; !ok {
		registry[name] = func(arguments map[string]any) (Tool, error) {
			raw, err := json.Marshal(arguments)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal parameters: %v", err)
			}

			if err := json.Unmarshal(raw, paramPtr.Interface()); err != nil {
				return nil, err
			}

			if v, ok := paramPtr.Interface().(ToolParameters); ok {
				if err := v.Validate(); err != nil {
					return nil, err
				}
			}

			field.Set(paramPtr.Elem())

			return tool, nil
		}
	}

	logger.Debug("RegisterTool completed")

	return llm.NewFunctionToolInfo(name, description, paramPtr.Interface()), nil
}
