package tool

import (
	"context"
	"reflect"

	"github.com/google/uuid"
)

type Tool interface {
	Parameters() reflect.Type

	Name() string
	Description() string

	Call(ctx context.Context, params ToolParameters) (ToolResponse, error)
}

type ToolParameters interface {
	Validate(tool Tool) error
	String() string
}

type ToolResponse interface {
	String() string
	Image() uuid.UUID
}
