package knowledge

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type embeddingWithScore struct {
	Embedding
	Score float32 `bun:"score"`
}

type EmbeddingRepository struct {
	db     *bun.DB
	logger *slog.Logger
}

func NewEmbeddingRepository(db *bun.DB) *EmbeddingRepository {
	return &EmbeddingRepository{
		db:     db,
		logger: slog.Default(),
	}
}

func (r *EmbeddingRepository) Get(ctx context.Context, id uuid.UUID) (*Embedding, error) {
	embedding := new(Embedding)
	err := r.db.NewSelect().
		Model(embedding).
		Where("e.id = ?", id).
		Relation("Chunk").
		Relation("Chunk.Knowledge").
		Relation("Chunk.Knowledge.Source").
		Limit(1).
		Scan(ctx)
	if err != nil {
		return nil, &Error{
			Code:    "EMBEDDING_NOT_FOUND",
			Message: "embedding not found",
			Err:     fmt.Errorf("embedding not found: %w", err),
		}
	}

	return embedding, nil
}

func (r *EmbeddingRepository) Create(ctx context.Context, embedding *Embedding) (uuid.UUID, error) {
	_, err := r.db.NewInsert().
		Model(embedding).
		Exec(ctx)
	if err != nil {
		return uuid.Nil, &Error{
			Code:    "EMBEDDING_INSERT_FAILED",
			Message: "failed to insert embedding into database",
			Err:     fmt.Errorf("failed to insert embedding into database: %w", err),
		}
	}

	return embedding.ID, nil
}

func (r *EmbeddingRepository) Similar(ctx context.Context, embedding []float32, limit int) ([]*Embedding, error) {
	var embeddingsWithScore []embeddingWithScore
	err := r.db.NewSelect().
		Model((*Embedding)(nil)).
		ColumnExpr("1 - (embedding <=> ?) AS score", embedding).
		Relation("Chunk").
		Relation("Chunk.Knowledge").
		Relation("Chunk.Knowledge.Source").
		OrderExpr("score DESC").
		Limit(limit).
		Scan(ctx, &embeddingsWithScore)

	if err != nil {
		return nil, &Error{
			Code:    "SIMILARITY_SEARCH_FAILED",
			Message: "failed to perform similarity search",
			Err:     fmt.Errorf("failed to perform similarity search: %w", err),
		}
	}

	var embeddings []*Embedding
	for _, e := range embeddingsWithScore {
		embeddings = append(embeddings, &e.Embedding)
	}

	return embeddings, nil
}
