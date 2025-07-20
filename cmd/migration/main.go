package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/migrations"
	"github.com/uptrace/bun/extra/bundebug"
	"github.com/uptrace/bun/migrate"
)

type Subcommand struct {
	Run         func(migrator *migrate.Migrator, name string, args []string) error
	Description string
}

var Subcommands = map[string]Subcommand{
	"init": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			return migrator.Init(c)
		},
		Description: "create migration tables",
	},
	"migrate": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			if err := migrator.Lock(c); err != nil {
				return err
			}
			defer migrator.Unlock(c) //nolint:errcheck

			group, err := migrator.Migrate(c)
			if err != nil {
				return err
			}
			if group.IsZero() {
				fmt.Printf("there are no new migrations to run (database is up to date)\n")
				return nil
			}
			fmt.Printf("migrated to %s\n", group)
			return nil
		},
		Description: "migrate database",
	},
	"rollback": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			if err := migrator.Lock(c); err != nil {
				return err
			}
			defer migrator.Unlock(c) //nolint:errcheck

			group, err := migrator.Rollback(c)
			if err != nil {
				return err
			}
			if group.IsZero() {
				fmt.Printf("there are no groups to roll back\n")
				return nil
			}
			fmt.Printf("rolled back %s\n", group)
			return nil
		},
		Description: "rollback the last migration group",
	},
	"lock": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			return migrator.Lock(c)
		},
		Description: "lock migrations",
	},
	"unlock": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			return migrator.Unlock(c)
		},
		Description: "unlock migrations",
	},
	"create_go": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			subFlag := flag.NewFlagSet(name, flag.ExitOnError)
			namePtr := subFlag.String("name", "", "migration name")

			err := subFlag.Parse(args)
			if err == flag.ErrHelp {
				return nil
			}

			if err != nil {
				return fmt.Errorf("failed to parse flags: %w", err)
			}

			if *namePtr == "" {
				subFlag.Usage()
				return fmt.Errorf("migration name is required")
			}

			mf, err := migrator.CreateGoMigration(c, *namePtr)
			if err != nil {
				return err
			}
			fmt.Printf("created migration %s (%s)\n", mf.Name, mf.Path)
			return nil
		},
		Description: "create Go migration",
	},
	"create_sql": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			subFlag := flag.NewFlagSet(name, flag.ExitOnError)
			namePtr := subFlag.String("name", "", "migration name")

			err := subFlag.Parse(args)
			if err == flag.ErrHelp {
				return nil
			}

			if err != nil {
				subFlag.Usage()
				return fmt.Errorf("failed to parse flags: %w", err)
			}

			if *namePtr == "" {
				return fmt.Errorf("migration name is required")
			}

			files, err := migrator.CreateSQLMigrations(c, *namePtr)
			if err != nil {
				return err
			}

			for _, mf := range files {
				fmt.Printf("created migration %s (%s)\n", mf.Name, mf.Path)
			}

			return nil
		},
		Description: "create up and down SQL migrations",
	},
	"create_tx_sql": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			subFlag := flag.NewFlagSet(name, flag.ExitOnError)
			namePtr := subFlag.String("name", "", "migration name")

			err := subFlag.Parse(args)
			if err == flag.ErrHelp {
				return nil
			}

			if err != nil {
				subFlag.Usage()
				return fmt.Errorf("failed to parse flags: %w", err)
			}

			if *namePtr == "" {
				return fmt.Errorf("migration name is required")
			}

			files, err := migrator.CreateTxSQLMigrations(c, *namePtr)
			if err != nil {
				return err
			}

			for _, mf := range files {
				fmt.Printf("created transaction migration %s (%s)\n", mf.Name, mf.Path)
			}

			return nil
		},
		Description: "create up and down transactional SQL migrations",
	},
	"status": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			ms, err := migrator.MigrationsWithStatus(c)
			if err != nil {
				return err
			}
			fmt.Printf("migrations: %s\n", ms)
			fmt.Printf("unapplied migrations: %s\n", ms.Unapplied())
			fmt.Printf("last migration group: %s\n", ms.LastGroup())
			return nil
		},
		Description: "print migrations status",
	},
	"mark_applied": {
		Run: func(migrator *migrate.Migrator, name string, args []string) error {
			c := context.Background()

			group, err := migrator.Migrate(c, migrate.WithNopMigration())
			if err != nil {
				return err
			}
			if group.IsZero() {
				fmt.Printf("there are no new migrations to mark as applied\n")
				return nil
			}
			fmt.Printf("marked as applied %s\n", group)
			return nil
		},
		Description: "mark migrations as applied without actually running them",
	},
}

func usage() {
	fmt.Println("Usage: scufris <subcommand> [args]")
	fmt.Println("Available subcommands:")
	for name, subcommand := range Subcommands {
		fmt.Printf("  %s: %s\n", name, subcommand.Description)
	}
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}

	cfg, err := config.LoadConfig()
	if err != nil {
		panic(fmt.Errorf("failed to load config: %w", err))
	}

	db := config.GetDB(cfg)

	db.AddQueryHook(bundebug.NewQueryHook(
		bundebug.WithEnabled(true),
		bundebug.FromEnv(),
	))

	templateData := map[string]string{
		"Prefix": "example_",
	}
	migrator := migrate.NewMigrator(db, migrations.Migrations, migrate.WithTemplateData(templateData))

	name := os.Args[1]
	args := os.Args[2:]
	subcommand, ok := Subcommands[name]
	if !ok {
		usage()
		fmt.Printf("ERROR: Unknown subcommand %s\n", name)
		os.Exit(1)
	}
	err = subcommand.Run(migrator, name, args)
	if err != nil {
		fmt.Printf("ERROR: %s\n", err)
		os.Exit(1)
	}
}
