package knowledge

import (
	"context"
	"log/slog"

	"github.com/alexjercan/scufris/registry"
	"github.com/google/uuid"
)

type TextOptions struct {
	Source string `json:"source" jsonschema:"title=source,description=The source of the text."`
}

type KnowledgeRegistry struct {
	imageRepository *ImageRepository
	embeddingRepository *EmbeddingRepository
	chunkRepository *ChunkRepository
	knowledgeRepository *KnowledgeRepository
	sourceRepository *KnowledgeSourceRepository

	logger *slog.Logger
}

func NewKnowledgeRegistry(
	imageRepository *ImageRepository,
	embeddingRepository *EmbeddingRepository,
	chunkRepository *ChunkRepository,
	knowledgeRepository *KnowledgeRepository,
	sourceRepository *KnowledgeSourceRepository,
) registry.Registry {
	return &KnowledgeRegistry{
		imageRepository:     imageRepository,
		embeddingRepository: embeddingRepository,
		chunkRepository:     chunkRepository,
		knowledgeRepository: knowledgeRepository,
		sourceRepository:    sourceRepository,
		logger:              slog.Default(),
	}
}

func (r *KnowledgeRegistry) AddText(ctx context.Context, text string, opts registry.TextOptions) (uuid.UUID, error) {
	// TODO: Implement text addition logic

	return uuid.Nil, nil
}

func (r *KnowledgeRegistry) GetText(ctx context.Context, id uuid.UUID) (string, error) {
	// TODO: Implement text retrieval logic

	return "", nil
}

func (r *KnowledgeRegistry) SearchText(ctx context.Context, query string, limit int, opts registry.TextOptions) ([]uuid.UUID, error) {
	// TODO: Implement text search logic
	return nil, nil
	/*
func (r *Retriever) Retrieve(ctx context.Context, req RetrieverRequest) ([]*Embedding, error) {
	r.logger.Debug("Retriever.Retrieve called",
		slog.String("query", req.Query),
		slog.Int("limit", req.Limit),
	)

	result, err := r.llm.Embeddings(ctx, llm.NewEmbeddingsRequest(r.model, req.Query))
	if err != nil {
		return nil, err
	}

	embedding := result.Embeddings[0]

	embeddings, err := r.repository.Similar(ctx, embedding, req.Limit)
	if err != nil {
		return nil, err
	}

	r.logger.Debug("Retriever.Retrieve completed",
		slog.Int("num_chunks", len(embeddings)),
		slog.String("query", req.Query),
		slog.Int("limit", req.Limit),
	)

	return embeddings, nil
}
*/
}

func (r *KnowledgeRegistry) AddImage(ctx context.Context, data string, opts registry.ImageOptions) (uuid.UUID, error) {
	// TODO: Implement image addition logic
	return uuid.Nil, nil
}

func (r *KnowledgeRegistry) GetImage(ctx context.Context, id uuid.UUID) (string, error) {
	// TODO: Implement image retrieval logic
	return "", nil
}
