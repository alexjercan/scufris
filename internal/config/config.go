package config

import (
	"database/sql"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/sqlitedialect"
	"github.com/uptrace/bun/driver/sqliteshim"
)

const EMBEDDING_MODEL = "nomic-embed-text"

const OLLAMA_URL = "http://localhost:11434"
const IMAGEGEN_URL = "http://localhost:8080"

const DATABASE_URL = "file:test.sqlite?cache=shared"

func GetDB() *bun.DB {
	sqldb, err := sql.Open(sqliteshim.ShimName, DATABASE_URL)
	if err != nil {
		panic(err)
	}

	return bun.NewDB(sqldb, sqlitedialect.New())
}
