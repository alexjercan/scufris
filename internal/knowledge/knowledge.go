package knowledge

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris"
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type KnowledgeRepository struct {
	db     *bun.DB
	logger *slog.Logger
}

func NewKnowledgeRepository(db *bun.DB) *KnowledgeRepository {
	return &KnowledgeRepository{
		db:     db,
		logger: slog.Default(),
	}
}

func (r *KnowledgeRepository) Get(ctx context.Context, id uuid.UUID) (*Knowledge, error) {
	knowledge := new(Knowledge)
	err := r.db.NewSelect().
		Model(knowledge).
		Where("k.id = ?", id).
		Relation("Source").
		Limit(1).
		Scan(ctx)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "KNOWLEDGE_NOT_FOUND",
			Message: "knowledge not found",
			Err:     fmt.Errorf("knowledge not found: %w", err),
		}
	}

	return knowledge, nil
}

func (r *KnowledgeRepository) Create(ctx context.Context, knowledge *Knowledge) (uuid.UUID, error) {
	_, err := r.db.NewInsert().
		Model(knowledge).
		Exec(ctx)
	if err != nil {
		return uuid.Nil, &scufris.Error{
			Code:    "KNOWLEDGE_INSERT_FAILED",
			Message: "failed to insert knowledge into database",
			Err:     fmt.Errorf("failed to insert knowledge into database: %w", err),
		}
	}

	return knowledge.ID, nil
}
