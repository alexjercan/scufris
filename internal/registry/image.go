package registry

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris"
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type ImageRegistry interface {
	AddImage(ctx context.Context, data string) (string, error)
	GetImage(ctx context.Context, id string) (string, error)
}

type MapImageRegistry struct {
	registry map[string]string
	logger   *slog.Logger
}

func NewMapImageRegistry() ImageRegistry {
	return &MapImageRegistry{
		registry: make(map[string]string),
		logger:   slog.Default(),
	}
}

func (r *MapImageRegistry) AddImage(ctx context.Context, data string) (string, error) {
	imageId := fmt.Sprintf("image_%d", len(r.registry))
	r.registry[imageId] = data

	r.logger.Debug("ImageRegistry.AddImage called",
		slog.String("imageId", imageId),
		slog.Int("totalImages", len(r.registry)),
	)

	return imageId, nil
}

func (r *MapImageRegistry) GetImage(ctx context.Context, id string) (string, error) {
	r.logger.Debug("ImageRegistry.GetImage called",
		slog.String("imageId", id),
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

type DbImageRegistry struct {
	db     *bun.DB
	logger *slog.Logger
}

func NewDbImageRegistry(db *bun.DB) ImageRegistry {
	return &DbImageRegistry{
		db:     db,
		logger: slog.Default(),
	}
}

func (r *DbImageRegistry) AddImage(ctx context.Context, data string) (string, error) {
	image := &Image{
		ID:   uuid.New(),
		Blob: data,
	}

	r.logger.Debug("ImageRegistry.AddImage called",
		slog.String("imageId", image.ID.String()),
	)

	_, err := r.db.NewInsert().Model(image).Exec(ctx)
	if err != nil {
		return "", &scufris.Error{
			Code:    "IMAGE_REGISTRY_ERROR",
			Message: "failed to add image to registry",
			Err:     fmt.Errorf("failed to add image to registry: %w", err),
		}
	}

	r.logger.Debug("ImageRegistry.AddImage successful",
		slog.String("imageId", image.ID.String()),
	)

	return image.ID.String(), nil
}

func (r *DbImageRegistry) GetImage(ctx context.Context, id string) (string, error) {
	r.logger.Debug("ImageRegistry.GetImage called",
		slog.String("imageId", id),
	)

	image := &Image{}
	err := r.db.NewSelect().Model(image).Where("id = ?", id).Scan(ctx)
	if err != nil {
		return "", &scufris.Error{
			Code:    "IMAGE_REGISTRY_ERROR",
			Message: "failed to get image from registry",
			Err:     fmt.Errorf("failed to get image from registry: %w", err),
		}
	}

	return image.Blob, nil
}

var defaultImageRegistry = NewMapImageRegistry()

type imageRegistryKeyType struct{}

var imageRegistryKey = imageRegistryKeyType{}

func WithImageRegistry(ctx context.Context, registry ImageRegistry) context.Context {
	return context.WithValue(ctx, imageRegistryKey, registry)
}

func AddImage(ctx context.Context, data string) (string, error) {
	registry, ok := ctx.Value(imageRegistryKey).(ImageRegistry)
	if !ok {
		registry = defaultImageRegistry
	}

	return registry.AddImage(ctx, data)
}

func GetImage(ctx context.Context, id string) (string, error) {
	registry, ok := ctx.Value(imageRegistryKey).(ImageRegistry)
	if !ok {
		registry = defaultImageRegistry
	}

	return registry.GetImage(ctx, id)
}
