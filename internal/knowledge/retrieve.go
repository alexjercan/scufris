package knowledge

import (
	"context"
	"log/slog"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/llm"
	"github.com/uptrace/bun"
)

type RetrieverRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

func NewRetrieverRequest(query string, limit int) *RetrieverRequest {
	return &RetrieverRequest{
		Query: query,
		Limit: limit,
	}
}

type Retriever struct {
	db     *bun.DB
	logger *slog.Logger
	model  string
	llm    llm.Llm
}

func NewRetriever(db *bun.DB, model string, client llm.Llm) *Retriever {
	return &Retriever{
		db:     db,
		logger: slog.Default(),
		model:  model,
		llm:    client,
	}
}

func (r *Retriever) Retrieve(ctx context.Context, req RetrieverRequest) ([]Embedding, error) {
	r.logger.Debug("Retriever.Retrieve called",
		slog.String("query", req.Query),
		slog.Int("limit", req.Limit),
	)

	result, err := r.llm.Embeddings(ctx, llm.NewEmbeddingsRequest(r.model, req.Query))
	if err != nil {
		return nil, &scufris.Error{
			Code:    "LLM_EMBEDDINGS_ERROR",
			Message: "failed to get embeddings from LLM",
			Err:     err,
		}
	}

	embedding := result.Embeddings[0]

	var embeddings []Embedding
	err = r.db.NewSelect().
		Model(&embeddings).
		ColumnExpr("1 - (embedding <=> ?) AS score", embedding).
		Relation("Chunk").
		OrderExpr("score DESC").
		Limit(req.Limit).
		Scan(ctx)

	if err != nil {
		return nil, &scufris.Error{
			Code:    "DB_RETRIEVE_ERROR",
			Message: "failed to retrieve chunks from database",
			Err:     err,
		}
	}

	return embeddings, nil
}
