package knowledge

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type ImageRepository struct {
	db     *bun.DB
	logger *slog.Logger
}

func NewImageRepository(db *bun.DB) *ImageRepository {
	return &ImageRepository{
		db:     db,
		logger: slog.Default(),
	}
}

func (r *ImageRepository) Get(ctx context.Context, id uuid.UUID) (*Image, error) {
	image := new(Image)
	err := r.db.NewSelect().
		Model(image).
		Where("i.id = ?", id).
		Relation("Knowledge").
		Relation("Knowledge.Source").
		Limit(1).
		Scan(ctx)
	if err != nil {
		return nil, &Error{
			Code:    "IMAGE_NOT_FOUND",
			Message: "image not found",
			Err:     fmt.Errorf("image not found: %w", err),
		}
	}

	return image, nil
}

func (r *ImageRepository) Create(ctx context.Context, image *Image) (uuid.UUID, error) {
	_, err := r.db.NewInsert().
		Model(image).
		Exec(ctx)
	if err != nil {
		return uuid.Nil, &Error{
			Code:    "IMAGE_INSERT_FAILED",
			Message: "failed to insert image into database",
			Err:     fmt.Errorf("failed to insert image into database: %w", err),
		}
	}

	return image.ID, nil
}
