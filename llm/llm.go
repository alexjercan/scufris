package llm

import (
	"context"
	"fmt"
)

type Llm interface {
	Chat(ctx context.Context, request ChatRequest) (ChatResponse, error)
}

type LlmError struct {
	Code    string
	Message string
	Err     error
}

func (e *LlmError) Error() string {
	return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Err)
}

func (e *LlmError) Unwrap() error {
	return e.Err
}
