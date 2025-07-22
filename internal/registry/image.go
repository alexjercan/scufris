package registry

import (
	"context"

	"github.com/google/uuid"
)

type ImageRegistry interface {
	AddImage(ctx context.Context, data string) (uuid.UUID, error)
	GetImage(ctx context.Context, id uuid.UUID) (string, error)
}
