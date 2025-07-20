package knowledge

import (
	"context"
	"log/slog"

	"github.com/alexjercan/scufris/llm"
	"github.com/google/uuid"
	"github.com/uptrace/bun"
)

type KnowledgeCommand int

const (
	CREATE KnowledgeCommand = iota
	UPDATE
	DELETE
)

type KnowledgeChanItem struct {
	Command     KnowledgeCommand
	KnowledgeID uuid.UUID
}

type KnowledgeWorker struct {
	db     *bun.DB
	ch     <-chan KnowledgeChanItem
	llm    llm.Llm
	logger *slog.Logger
	model  string
}

func NewKnowledgeWorker(db *bun.DB, ch <-chan KnowledgeChanItem, model string, llm llm.Llm) *KnowledgeWorker {
	return &KnowledgeWorker{
		db:     db,
		ch:     ch,
		llm:    llm,
		logger: slog.Default(),
		model:  model,
	}
}

func (w *KnowledgeWorker) Start(ctx context.Context) {
	for item := range w.ch {
		switch item.Command {
		case CREATE:
			w.handleCreate(ctx, item.KnowledgeID)
		case UPDATE:
			// w.handleUpdate(ctx, item.KnowledgeID)
		case DELETE:
			// w.handleDelete(ctx, item.KnowledgeID)
		}
	}
}

func (w *KnowledgeWorker) handleCreate(ctx context.Context, knowledgeID uuid.UUID) {
	w.logger.Debug("Worker.handleCreate called",
		slog.String("knowledgeID", knowledgeID.String()),
		slog.String("model", w.model),
	)

	k := new(Knowledge)
	err := w.db.NewSelect().Model(k).Where("id = ?", knowledgeID).Scan(ctx)
	if err != nil {
		w.logger.Error("Failed to fetch knowledge for creation",
			slog.String("knowledgeID", knowledgeID.String()),
			slog.Any("error", err),
		)

		// TODO: Handle error appropriately, maybe retry or log more details
		return
	}

	chunk := NewChunk(knowledgeID, 0, k.Content)
	_, err = w.db.NewInsert().Model(chunk).Exec(ctx)
	if err != nil {
		w.logger.Error("Failed to insert chunk for knowledge",
			slog.String("knowledgeID", knowledgeID.String()),
			slog.Any("error", err),
		)

		// TODO: Handle error appropriately, maybe retry or log more details
		return
	}

	result, err := w.llm.Embeddings(ctx, llm.NewEmbeddingsRequest(w.model, k.Content))
	if err != nil {
		w.logger.Error("Failed to generate embeddings for knowledge",
			slog.String("knowledgeID", knowledgeID.String()),
			slog.Any("error", err),
		)

		// TODO: Handle error appropriately, maybe retry or log more details
		return
	}

	embedding := NewEmbedding(chunk.ID, result.Embeddings[0])
	_, err = w.db.NewInsert().Model(embedding).Exec(ctx)
	if err != nil {
		w.logger.Error("Failed to insert embedding for knowledge",
			slog.String("knowledgeID", knowledgeID.String()),
			slog.Any("error", err),
		)

		// TODO: Handle error appropriately, maybe retry or log more details
		return
	}

	w.logger.Debug("Knowledge created and processed successfully",
		slog.String("knowledgeID", knowledgeID.String()),
		slog.String("chunkID", chunk.ID.String()),
		slog.String("embeddingID", embedding.ID.String()),
	)
}
