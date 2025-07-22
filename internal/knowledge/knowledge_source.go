package knowledge

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris"
	"github.com/uptrace/bun"
)

type KnowledgeSourceRepository struct {
	db     *bun.DB
	logger *slog.Logger
}

func NewKnowledgeSourceRepository(db *bun.DB) *KnowledgeSourceRepository {
	return &KnowledgeSourceRepository{
		db:     db,
		logger: slog.Default(),
	}
}

func (r *KnowledgeSourceRepository) GetByName(ctx context.Context, name string) (*KnowledgeSource, error) {
	source := new(KnowledgeSource)
	err := r.db.NewSelect().Model(source).Where("ks.name = ?", name).Limit(1).Scan(ctx)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "SOURCE_NOT_FOUND",
			Message: "knowledge source not found",
			Err:     fmt.Errorf("knowledge source not found: %w", err),
		}
	}

	return source, nil
}
