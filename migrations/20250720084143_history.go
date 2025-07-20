package migrations

import (
	"context"
	"fmt"

	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

func init() {
	Migrations.MustRegister(func(ctx context.Context, db *bun.DB) error {
		fmt.Print(" [up migration] ")

		_, err := db.NewRaw("CREATE EXTENSION IF NOT EXISTS vector").Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to create vector extension: %w", err)
		}

		_, err = db.NewCreateTable().Model((*knowledge.KnowledgeSource)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to create knowledge sources table: %w", err)
		}

		_, err = db.NewCreateTable().Model((*knowledge.Knowledge)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to create knowledge table: %w", err)
		}

		_, err = db.NewCreateTable().Model((*knowledge.Chunk)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to create knowledge chunks table: %w", err)
		}

		_, err = db.NewCreateTable().Model((*knowledge.Embedding)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to create knowledge embeddings table: %w", err)
		}

		source := &knowledge.KnowledgeSource{
			ID:          uuid.New(),
			Name:        "transcript",
			Description: "Transcript of previous conversations",
		}
		_, err = db.NewInsert().Model(source).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to insert initial knowledge source: %w", err)
		}

		return nil
	}, func(ctx context.Context, db *bun.DB) error {
		fmt.Print(" [down migration] ")

		_, err := db.NewDropTable().Model((*knowledge.Embedding)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to drop knowledge embeddings table: %w", err)
		}

		_, err = db.NewDropTable().Model((*knowledge.Chunk)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to drop knowledge chunks table: %w", err)
		}

		_, err = db.NewDropTable().Model((*knowledge.KnowledgeSource)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to drop knowledge sources table: %w", err)
		}

		_, err = db.NewDropTable().Model((*knowledge.Knowledge)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to drop knowledge table: %w", err)
		}

		_, err = db.NewRaw("DROP EXTENSION IF EXISTS vector").Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to drop vector extension: %w", err)
		}

		return nil
	})
}
