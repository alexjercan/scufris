package llm

import (
	"context"

	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/verbose"
)

type verboseLlm struct {
	llm Llm
}

func (l *verboseLlm) Chat(ctx context.Context, request ChatRequest) (ChatResponse, error) {
	resp, err := l.llm.Chat(ctx, request)
	if err != nil {
		return resp, err
	}

	if name, ok := contextkeys.AgentName(ctx); ok {
		if len(resp.Message.Content) > 0 {
			verbose.Say(name, resp.Message.Content)
		}
	}

	return resp, nil
}
