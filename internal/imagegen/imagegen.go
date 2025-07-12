package imagegen

import (
	"context"
)

type ImageGenerator interface {
	Generate(ctx context.Context, request GenerateRequest) ([]byte, error)
}
