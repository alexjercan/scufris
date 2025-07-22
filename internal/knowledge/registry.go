package knowledge

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/alexjercan/scufris/llm"
	"github.com/alexjercan/scufris/registry"
	"github.com/google/uuid"
)

type TextOptions struct {
	ChunkID uuid.UUID `json:"chunk_id" jsonschema:"title=chunk_id,description=The ID of the chunk to which this text belongs."`
}

type ImageOptions struct {
	KnowledgeID uuid.UUID `json:"knowledge_id" jsonschema:"title=knowledge_id,description=The ID of the knowledge to which this image belongs."`
}

type KnowledgeRegistry struct {
	model string
	client llm.Llm

	imageRepository *ImageRepository
	embeddingRepository *EmbeddingRepository
	chunkRepository *ChunkRepository
	knowledgeRepository *KnowledgeRepository
	sourceRepository *KnowledgeSourceRepository

	logger *slog.Logger
}

func NewKnowledgeRegistry(
	model string,
	client llm.Llm,
	imageRepository *ImageRepository,
	embeddingRepository *EmbeddingRepository,
	chunkRepository *ChunkRepository,
	knowledgeRepository *KnowledgeRepository,
	sourceRepository *KnowledgeSourceRepository,
) registry.Registry {
	return &KnowledgeRegistry{
		model:              model,
		client:             client,

		imageRepository:     imageRepository,
		embeddingRepository: embeddingRepository,
		chunkRepository:     chunkRepository,
		knowledgeRepository: knowledgeRepository,
		sourceRepository:    sourceRepository,

		logger:              slog.Default(),
	}
}

func (r *KnowledgeRegistry) AddText(ctx context.Context, text string, opts registry.TextOptions) (uuid.UUID, error) {
	r.logger.Debug("AddText called",
		slog.String("textSize", fmt.Sprintf("%d bytes", len(text))),
		slog.Any("options", opts),
	)

	o, ok := opts.(*TextOptions)
	if !ok {
		return uuid.Nil, &Error{
			Code:    "INVALID_OPTIONS",
			Message: "invalid options provided for AddText",
			Err:     fmt.Errorf("invalid options provided for AddText: %T", opts),
		}
	}
	chunkID := o.ChunkID

	response, err := r.client.Embeddings(ctx, llm.NewEmbeddingsRequest(r.model, text))
	if err != nil {
		return uuid.Nil, &Error{
			Code:    "EMBEDDING_ERROR",
			Message: "failed to create embedding for text",
			Err:     fmt.Errorf("failed to create embedding for text: %w", err),
		}
	}

	id, err := r.embeddingRepository.Create(ctx, NewEmbedding(chunkID, response.Embeddings[0], text))
	if err != nil {
		return uuid.Nil, &Error{
			Code:    "EMBEDDING_CREATION_ERROR",
			Message: "failed to create embedding in database",
			Err:     fmt.Errorf("failed to create embedding in database: %w", err),
		}
	}

	r.logger.Debug("AddText completed",
		slog.String("textSize", fmt.Sprintf("%d bytes", len(text))),
		slog.String("embeddingID", id.String()),
		slog.Any("options", opts),
	)

	return id, nil
}

func (r *KnowledgeRegistry) GetText(ctx context.Context, id uuid.UUID) (string, error) {
	r.logger.Debug("GetText called",
		slog.String("id", id.String()),
	)

	embedding, err := r.embeddingRepository.Get(ctx, id)
	if err != nil {
		return "", &Error{
			Code:    "TEXT_RETRIEVAL_ERROR",
			Message: "failed to retrieve text from database",
			Err:     fmt.Errorf("failed to retrieve text from database: %w", err),
		}
	}

	r.logger.Debug("GetText completed",
		slog.String("id", id.String()),
		slog.Int("textSize", len(embedding.Content)),
	)

	return embedding.Content, nil
}

func (r *KnowledgeRegistry) SearchText(ctx context.Context, query string, limit int, opts registry.TextOptions) ([]uuid.UUID, error) {
	r.logger.Debug("SearchText called",
		slog.String("query", query),
		slog.Int("limit", limit),
		slog.Any("options", opts),
	)

	response, err := r.client.Embeddings(ctx, llm.NewEmbeddingsRequest(r.model, query))
	if err != nil {
		return nil, &Error{
			Code:    "EMBEDDING_ERROR",
			Message: "failed to create embedding for query",
			Err:     fmt.Errorf("failed to create embedding for query: %w", err),
		}
	}

	embeddings, err := r.embeddingRepository.Similar(ctx, response.Embeddings[0], limit)
	if err != nil {
		return nil, &Error{
			Code:    "SIMILARITY_SEARCH_ERROR",
			Message: "failed to perform similarity search",
			Err:     fmt.Errorf("failed to perform similarity search: %w", err),
		}
	}

	r.logger.Debug("SearchText completed",
		slog.String("query", query),
		slog.Int("limit", limit),
		slog.Int("results_count", len(embeddings)),
		slog.Any("options", opts),
	)

	ids := make([]uuid.UUID, 0, len(embeddings))
	for _, embedding := range embeddings {
		ids = append(ids, embedding.ID)
	}

	return ids, nil
}

func (r *KnowledgeRegistry) AddImage(ctx context.Context, data string, opts registry.ImageOptions) (uuid.UUID, error) {
	r.logger.Debug("AddImage called",
		slog.String("dataSize", fmt.Sprintf("%d bytes", len(data))),
		slog.Any("options", opts),
	)

	o, ok := opts.(*ImageOptions)
	if !ok {
		return uuid.Nil, &Error{
			Code:    "INVALID_OPTIONS",
			Message: "invalid options provided for AddImage",
			Err:     fmt.Errorf("invalid options provided for AddImage: %T", opts),
		}
	}
	knowledgeID := o.KnowledgeID

	id, err := r.imageRepository.Create(ctx, NewImage(knowledgeID, data))
	if err != nil {
		return uuid.Nil, &Error{
			Code:    "IMAGE_CREATION_ERROR",
			Message: "failed to create image in database",
			Err:     fmt.Errorf("failed to create image in database: %w", err),
		}
	}

	r.logger.Debug("AddImage completed",
		slog.String("dataSize", fmt.Sprintf("%d bytes", len(data))),
		slog.String("imageID", id.String()),
		slog.Any("options", opts),
	)

	return id, nil
}

func (r *KnowledgeRegistry) GetImage(ctx context.Context, id uuid.UUID) (string, error) {
	r.logger.Debug("GetImage called",
		slog.String("id", id.String()),
	)

	image, err := r.imageRepository.Get(ctx, id)
	if err != nil {
		return "", &Error{
			Code:    "IMAGE_RETRIEVAL_ERROR",
			Message: "failed to retrieve image from database",
			Err:     fmt.Errorf("failed to retrieve image from database: %w", err),
		}
	}

	r.logger.Debug("GetImage completed",
		slog.String("id", id.String()),
		slog.Int("imageSize", len(image.Blob)),
	)

	return image.Blob, nil
}
