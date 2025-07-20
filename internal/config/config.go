package config

import (
	"database/sql"
	"fmt"

	"github.com/ilyakaznacheev/cleanenv"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"
	"github.com/uptrace/bun/extra/bundebug"
)

// Somehow make these variables so I can set them from nix
type Config struct {
	ConfigPath string `env:"CONFIG_PATH" env-default:"config.yaml"`
	Database   struct {
		Host     string `yaml:"host"`
		Port     int    `yaml:"port"`
		User     string `yaml:"user"`
		Password string `yaml:"password"`
		Database string `yaml:"database"`
		Insecure bool   `yaml:"insecure"`
	} `yaml:"database"`
	Ollama struct {
		Url   string `yaml:"url"`
		Model string `yaml:"model"`
	} `yaml:"ollama"`
	ImageGen struct {
		Url string `yaml:"url"`
	} `yaml:"imagegen"`
	EmbeddingModel string `yaml:"embedding_model"`
	SocketPath     string `yaml:"socket_path"`
}

func LoadConfig() (cfg Config, err error) {
	err = cleanenv.ReadEnv(&cfg)
	if err != nil {
		return
	}

	err = cleanenv.ReadConfig(cfg.ConfigPath, &cfg)
	if err != nil {
		return
	}

	return
}

func GetDB(cfg Config) *bun.DB {
	pgconn := pgdriver.NewConnector(
		pgdriver.WithAddr(fmt.Sprintf("%s:%d", cfg.Database.Host, cfg.Database.Port)),
		pgdriver.WithUser(cfg.Database.User),
		pgdriver.WithPassword(cfg.Database.Password),
		pgdriver.WithDatabase(cfg.Database.Database),
		pgdriver.WithInsecure(cfg.Database.Insecure),
	)

	sqldb := sql.OpenDB(pgconn)
	db := bun.NewDB(sqldb, pgdialect.New())
	db.AddQueryHook(bundebug.NewQueryHook(bundebug.WithVerbose(false)))

	return db
}
