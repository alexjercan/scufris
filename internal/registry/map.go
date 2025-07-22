package registry

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris"
	"github.com/google/uuid"
)

type MapImageRegistry struct {
	registry map[uuid.UUID]string
	logger   *slog.Logger
}

func NewMapImageRegistry() ImageRegistry {
	return &MapImageRegistry{
		registry: make(map[uuid.UUID]string),
		logger:   slog.Default(),
	}
}

func (r *MapImageRegistry) AddImage(ctx context.Context, data string) (uuid.UUID, error) {
	imageId := uuid.New()
	r.registry[imageId] = data

	r.logger.Debug("ImageRegistry.AddImage called",
		slog.String("imageId", imageId.String()),
		slog.Int("totalImages", len(r.registry)),
	)

	return imageId, nil
}

func (r *MapImageRegistry) GetImage(ctx context.Context, id uuid.UUID) (string, error) {
	r.logger.Debug("ImageRegistry.GetImage called",
		slog.String("imageId", id.String()),
	)

	img, ok := r.registry[id]
	if !ok {
		return "", &scufris.Error{
			Code:    "IMAGE_NOT_FOUND",
			Message: "image not found in registry",
			Err:     fmt.Errorf("image with id %s not found in registry", id),
		}
	}

	return img, nil
}
