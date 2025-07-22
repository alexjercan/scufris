package llm

import (
	"context"
)

type Llm interface {
	Chat(ctx context.Context, request ChatRequest, opts *ChatOptions) (response ChatResponse, err error)
	Embeddings(ctx context.Context, request EmbeddingsRequest) (EmbeddingsResponse, error)
	ModelInfo(ctx context.Context, model string) (ModelInfoResponse, error)
}
