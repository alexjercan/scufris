package knowledge

import (
	"context"
	"log/slog"

	"github.com/alexjercan/scufris/llm"
)

type RetrieverRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

func NewRetrieverRequest(query string, limit int) RetrieverRequest {
	return RetrieverRequest{
		Query: query,
		Limit: limit,
	}
}

type Retriever struct {
	repository *EmbeddingRepository
	model      string
	llm        llm.Llm
	logger     *slog.Logger
}

func NewRetriever(repository *EmbeddingRepository, model string, client llm.Llm) *Retriever {
	return &Retriever{
		repository: repository,
		model:      model,
		llm:        client,
		logger:     slog.Default(),
	}
}

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
