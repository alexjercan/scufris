package llm

import (
	"context"
)

type Llm interface {
	Chat(ctx context.Context, request ChatRequest) (ChatResponse, error)
	Embeddings(ctx context.Context, request EmbeddingsRequest) (EmbeddingsResponse, error)
}
