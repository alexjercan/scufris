package llm

import (
	"context"
)

type Llm interface {
	Chat(ctx context.Context, request ChatRequest) (ChatResponse, error)
}
