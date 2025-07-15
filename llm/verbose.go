package llm

import (
	"context"

	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/verbose"
)

type verboseLlm struct {
	llm Llm
}

func (l *verboseLlm) Chat(ctx context.Context, request ChatRequest, onToken ChatOnToken) (ChatResponse, error) {
	name, ok := contextkeys.AgentName(ctx)

	if ok {
		verbose.SayStart(name)
		defer verbose.SayEnd()
	}

	resp, err := l.llm.Chat(ctx, request, func(s string) error {
		if ok {
			verbose.SayToken(s)
		}

		if onToken != nil {
			onToken(s)
		}

		return nil
	})

	return resp, err
}
