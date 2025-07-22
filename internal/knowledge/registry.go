package knowledge

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/google/uuid"
)

type KnowledgeImageRegistry struct {
	imageRepository *ImageRepository

	logger *slog.Logger
}

func NewKnowledgeImageRegistry(imageRepository *ImageRepository) registry.ImageRegistry {
	return &KnowledgeImageRegistry{
		imageRepository: imageRepository,
		logger:          slog.Default(),
	}
}

func (r *KnowledgeImageRegistry) AddImage(ctx context.Context, data string) (uuid.UUID, error) {
	r.logger.Debug("ImageRegistry.AddImage called",
		slog.String("dataSize", fmt.Sprintf("%d bytes", len(data))),
	)

	// TODO: Add knowledge ID to the image if needed
	image := NewImage(uuid.Nil, data)

	id, err := r.imageRepository.Create(ctx, image)
	if err != nil {
		return uuid.Nil, &scufris.Error{
			Code:    "IMAGE_REGISTRY_ERROR",
			Message: "failed to add image to registry",
			Err:     fmt.Errorf("failed to add image to registry: %w", err),
		}
	}

	r.logger.Debug("ImageRegistry.AddImage successful",
		slog.String("imageId", image.ID.String()),
	)

	return id, nil
}

func (r *KnowledgeImageRegistry) GetImage(ctx context.Context, id uuid.UUID) (string, error) {
	r.logger.Debug("ImageRegistry.GetImage called",
		slog.String("imageId", id.String()),
	)

	image, err := r.imageRepository.Get(ctx, id)
	if err != nil {
		return "", &scufris.Error{
			Code:    "IMAGE_REGISTRY_ERROR",
			Message: "failed to get image from registry",
			Err:     fmt.Errorf("failed to get image from registry: %w", err),
		}
	}

	return image.Blob, nil
}
