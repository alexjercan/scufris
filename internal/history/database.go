package history

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type DbTranscriptWriter struct {
	buffer bytes.Buffer
	db     *bun.DB
	logger *slog.Logger
	ch     chan<- knowledge.KnowledgeChanItem
}

func NewDbTranscriptWriter(db *bun.DB, ch chan<- knowledge.KnowledgeChanItem) TranscriptSink {
	return &DbTranscriptWriter{
		buffer: bytes.Buffer{},
		db:     db,
		logger: slog.Default(),
		ch:     ch,
	}
}

func (t *DbTranscriptWriter) Write(p []byte) (int, error) {
	return t.buffer.Write(p)
}

func (t *DbTranscriptWriter) Close() error {
	ctx := context.Background()

	source := new(knowledge.KnowledgeSource)
	err := t.db.NewSelect().Model(source).Where("name = ?", "transcript").Limit(1).Scan(ctx)
	if err != nil {
		return &scufris.Error{
			Code:    "TRANSCRIPT_SOURCE_NOT_FOUND",
			Message: "transcript source not found in database",
			Err:     fmt.Errorf("transcript source not found in database: %w", err),
		}
	}

	k := &knowledge.Knowledge{
		ID:       uuid.New(),
		SourceID: source.ID,
		Content:  t.buffer.String(),
	}

	_, err = t.db.NewInsert().Model(k).Exec(ctx)
	if err != nil {
		return &scufris.Error{
			Code:    "TRANSCRIPT_INSERT_FAILED",
			Message: "failed to insert transcript into database",
			Err:     fmt.Errorf("failed to insert transcript into database: %w", err),
		}
	}

	t.ch <- knowledge.KnowledgeChanItem{
		Command:     knowledge.CREATE,
		KnowledgeID: k.ID,
	}

	t.logger.Debug("Transcript written to database", slog.String("source_id", source.ID.String()), slog.Any("knowledge_id", k.ID))

	return nil
}
