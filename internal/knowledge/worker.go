package knowledge

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris/llm"
	"github.com/google/uuid"
)

// TODO: In case of error we should consider adding some kind of flag to the chunk or create a new table for chunk errors

type KnowledgeCommand interface {
	Execute(ctx context.Context) error
}

type CreateKnowledgeCommand struct {
	embeddingRepository *EmbeddingRepository
	model               string
	llm                 llm.Llm
	chunkID             uuid.UUID
	content             string

	logger *slog.Logger
}

func (c *CreateKnowledgeCommand) Execute(ctx context.Context) error {
	c.logger.Debug("CreateKnowledgeCommand.Execute called",
		slog.String("chunkID", c.chunkID.String()),
		slog.String("content", c.content),
		slog.String("model", c.model),
	)

	result, err := c.llm.Embeddings(ctx, llm.NewEmbeddingsRequest(c.model, c.content))
	if err != nil {
		return err
	}

	embedding := NewEmbedding(c.chunkID, result.Embeddings[0], c.content)
	_, err = c.embeddingRepository.Create(ctx, embedding)
	if err != nil {
		return err
	}

	c.logger.Debug("Knowledge created successfully",
		slog.String("chunkID", c.chunkID.String()),
		slog.String("embeddingID", embedding.ID.String()),
	)

	return nil
}

type KnowledgeCommandFactory struct {
	chunkRepository     *ChunkRepository
	embeddingRepository *EmbeddingRepository
	model               string
	llm                 llm.Llm
}

func NewKnowledgeCommandFactory(
	chunkRepository *ChunkRepository,
	embeddingRepository *EmbeddingRepository,
	model string,
	llm llm.Llm,
) *KnowledgeCommandFactory {
	return &KnowledgeCommandFactory{
		chunkRepository:     chunkRepository,
		embeddingRepository: embeddingRepository,
		model:               model,
		llm:                 llm,
	}
}

func (f *KnowledgeCommandFactory) NewCreateCommand(chunkID uuid.UUID, content string) KnowledgeCommand {
	return &CreateKnowledgeCommand{
		embeddingRepository: f.embeddingRepository,
		model:               f.model,
		llm:                 f.llm,
		chunkID:             chunkID,
		content:             content,
		logger:              slog.Default(),
	}
}

type KnowledgeWorker struct {
	ch     <-chan KnowledgeCommand
	logger *slog.Logger
}

func NewKnowledgeWorker(ch <-chan KnowledgeCommand) *KnowledgeWorker {
	return &KnowledgeWorker{
		ch:     ch,
		logger: slog.Default(),
	}
}

func (w *KnowledgeWorker) Start(ctx context.Context) {
	for item := range w.ch {
		w.logger.Debug("KnowledgeWorker processing command",
			slog.String("command_type", fmt.Sprintf("%T", item)),
		)

		if err := item.Execute(ctx); err != nil {
			w.logger.Error("Failed to execute knowledge command",
				slog.String("error", err.Error()),
			)
			continue
		}

		w.logger.Debug("Knowledge command executed successfully")
	}
}
