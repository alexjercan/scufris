package tools

import (
	"context"
	"reflect"
)

type Tool interface {
	Parameters() reflect.Type

	Name() string
	Description() string

	Call(ctx context.Context, params ToolParameters) (any, error)
}

type ToolParameters interface {
	Validate() error
}
