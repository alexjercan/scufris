package registry

import (
	"context"
	"fmt"
	"log/slog"
)

type ImageRegistry struct {
	registry map[string]string
	logger   *slog.Logger
}

func NewImageRegistry() *ImageRegistry {
	return &ImageRegistry{
		registry: make(map[string]string),
		logger:   slog.Default(),
	}
}

func (r *ImageRegistry) addImage(data string) string {
	imageId := fmt.Sprintf("image_%d", len(r.registry))
	r.registry[imageId] = data

	r.logger.Debug("ImageRegistry.AddImage called",
		slog.String("imageId", imageId),
		slog.Int("totalImages", len(r.registry)),
	)

	return imageId
}

func (r *ImageRegistry) getImage(id string) (string, bool) {
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

var defaultImageRegistry = NewImageRegistry()

type imageRegistryKeyType struct{}

var imageRegistryKey = imageRegistryKeyType{}

func WithImageRegistry(ctx context.Context, registry *ImageRegistry) context.Context {
	return context.WithValue(ctx, imageRegistryKey, registry)
}

func AddImage(ctx context.Context, data string) string {
	registry, ok := ctx.Value(imageRegistryKey).(*ImageRegistry)
	if !ok {
		registry = defaultImageRegistry
	}

	return registry.addImage(data)
}

func GetImage(ctx context.Context, id string) (string, bool) {
	registry, ok := ctx.Value(imageRegistryKey).(*ImageRegistry)
	if !ok {
		registry = defaultImageRegistry
	}

	return registry.getImage(id)
}
