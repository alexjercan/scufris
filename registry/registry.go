package registry

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"

	"github.com/alexjercan/scufris"
	"github.com/google/uuid"
)

type ImageOptions any
type TextOptions any

type Registry interface {
	AddText(ctx context.Context, text string, opts TextOptions) (uuid.UUID, error)
	GetText(ctx context.Context, id uuid.UUID) (string, error)
	SearchText(ctx context.Context, query string, limit int, opts TextOptions) ([]uuid.UUID, error)

	AddImage(ctx context.Context, data string, opts ImageOptions) (uuid.UUID, error)
	GetImage(ctx context.Context, id uuid.UUID) (string, error)
}

type MapTextOptions struct {
	Path string
}

type MapRegistry struct {
	images map[uuid.UUID]string
	texts  map[uuid.UUID]string

	logger *slog.Logger
}

func NewMapRegistry() Registry {
	return &MapRegistry{
		images: make(map[uuid.UUID]string),
		texts:  make(map[uuid.UUID]string),

		logger: slog.Default(),
	}
}

func (r *MapRegistry) AddText(ctx context.Context, text string, opts TextOptions) (uuid.UUID, error) {
	id := uuid.New()

	options := opts.(*MapTextOptions)
	if options != nil {
		r.logger.Debug("MapRegistry.AddText with path option",
			slog.String("path", options.Path),
			slog.String("id", id.String()),
		)

		if err := os.WriteFile(options.Path, []byte(text), 0644); err != nil {
			return uuid.Nil, &scufris.Error{
				Code:    "TEXT_WRITE_ERROR",
				Message: "failed to write text to file",
				Err:     fmt.Errorf("failed to write text to file %s: %w", options.Path, err),
			}
		}
	}

	r.logger.Debug("MapRegistry.AddText called",
		slog.String("size", fmt.Sprintf("%d bytes", len(text))),
		slog.String("id", id.String()),
	)

	r.texts[id] = text
	return id, nil
}

func (r *MapRegistry) GetText(ctx context.Context, id uuid.UUID) (string, error) {
	r.logger.Debug("MapRegistry.GetText called",
		slog.String("id", id.String()),
	)

	text, ok := r.texts[id]
	if !ok {
		return "", &scufris.Error{
			Code:    "TEXT_NOT_FOUND",
			Message: "text not found in registry",
			Err:     fmt.Errorf("text with id %s not found in registry", id),
		}
	}

	r.logger.Debug("MapRegistry.GetText successful",
		slog.String("id", id.String()),
		slog.String("size", fmt.Sprintf("%d bytes", len(text))),
	)

	return text, nil
}

func matchScore(text, query string) int {
	count := 0
	words := strings.Split(text, " ")
	queryWords := strings.Split(query, " ")
	for _, word := range words {
		if slices.Contains(queryWords, word) {
			count++
		}
	}

	return count
}

func (r *MapRegistry) SearchText(ctx context.Context, query string, limit int, opts TextOptions) ([]uuid.UUID, error) {
	r.logger.Debug("MapRegistry.SearchText called",
		slog.String("query", query),
		slog.Int("limit", limit),
	)

	ids := make([]uuid.UUID, 0, len(r.texts))
	for id := range r.texts {
		ids = append(ids, id)
	}

	slices.SortFunc(ids, func(a, b uuid.UUID) int {
		scoreA := matchScore(r.texts[a], query)
		scoreB := matchScore(r.texts[b], query)

		return scoreB - scoreA // Sort by score descending
	})

	result := make([]uuid.UUID, 0, limit)
	for i, id := range ids {
		if i >= limit {
			break
		}
		result = append(result, id)
	}

	r.logger.Debug("MapRegistry.SearchText successful",
		slog.String("query", query),
		slog.Int("resultCount", len(result)),
	)

	return result, nil
}

func (r *MapRegistry) AddImage(ctx context.Context, data string, opts ImageOptions) (uuid.UUID, error) {
	id := uuid.New()

	r.logger.Debug("MapRegistry.AddImage called",
		slog.String("dataSize", fmt.Sprintf("%d bytes", len(data))),
		slog.String("id", id.String()),
	)

	r.images[id] = data
	return id, nil
}

func (r *MapRegistry) GetImage(ctx context.Context, id uuid.UUID) (string, error) {
	r.logger.Debug("MapRegistry.GetImage called",
		slog.String("imageId", id.String()),
	)

	image, ok := r.images[id]
	if !ok {
		return "", &scufris.Error{
			Code:    "IMAGE_NOT_FOUND",
			Message: "image not found in registry",
			Err:     fmt.Errorf("image with id %s not found in registry", id),
		}
	}

	r.logger.Debug("MapRegistry.GetImage successful",
		slog.String("imageId", id.String()),
		slog.String("dataSize", fmt.Sprintf("%d bytes", len(image))),
	)

	return image, nil
}
