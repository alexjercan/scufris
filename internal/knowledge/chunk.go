package knowledge

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type ChunkRepository struct {
	db     *bun.DB
	logger *slog.Logger
}

func NewChunkRepository(db *bun.DB) *ChunkRepository {
	return &ChunkRepository{
		db:     db,
		logger: slog.Default(),
	}
}

func (r *ChunkRepository) Get(ctx context.Context, id uuid.UUID) (*Chunk, error) {
	chunk := new(Chunk)
	err := r.db.NewSelect().
		Model(chunk).
		Where("c.id = ?", id).
		Relation("Knowledge").
		Relation("Knowledge.Source").
		Limit(1).
		Scan(ctx)
	if err != nil {
		return nil, &Error{
			Code:    "CHUNK_NOT_FOUND",
			Message: "chunk not found",
			Err:     fmt.Errorf("chunk not found: %w", err),
		}
	}

	return chunk, nil
}

func (r *ChunkRepository) Create(ctx context.Context, chunk *Chunk) (uuid.UUID, error) {
	_, err := r.db.NewInsert().
		Model(chunk).
		Exec(ctx)
	if err != nil {
		return uuid.Nil, &Error{
			Code:    "CHUNK_INSERT_FAILED",
			Message: "failed to insert chunk into database",
			Err:     fmt.Errorf("failed to insert chunk into database: %w", err),
		}
	}

	return chunk.ID, nil
}
