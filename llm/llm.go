package llm

import (
	"context"
)

type ChatOnToken func(string) error

type Llm interface {
	Chat(ctx context.Context, request ChatRequest, onToken ChatOnToken) (ChatResponse, error)
}
