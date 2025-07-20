package migrations

import (
	"context"
	"fmt"

	"github.com/alexjercan/scufris/internal/registry"
	"github.com/uptrace/bun"
)

func init() {
	Migrations.MustRegister(func(ctx context.Context, db *bun.DB) error {
		fmt.Print(" [up migration] ")

		_, err := db.NewCreateTable().Model((*registry.Image)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to create images table: %w", err)
		}

		return nil
	}, func(ctx context.Context, db *bun.DB) error {
		fmt.Print(" [down migration] ")

		_, err := db.NewDropTable().Model((*registry.Image)(nil)).Exec(ctx)
		if err != nil {
			return fmt.Errorf("failed to drop images table: %w", err)
		}

		return nil
	})
}
