package history

import (
	"context"
	"fmt"

	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/observer"
)

type historyObserver struct {
	w TranscriptSink
}

func NewHistoryObserver(w TranscriptSink) observer.Observer {
	return &historyObserver{
		w: w,
	}
}

func (o *historyObserver) OnUser(ctx context.Context, message string) error {
	if _, err := o.w.Write([]byte("User: " + message + "\n")); err != nil {
		return err
	}

	return nil
}

func (o *historyObserver) OnStart(ctx context.Context) error {
	if name, ok := contextkeys.AgentName(ctx); ok {
		if _, err := o.w.Write([]byte(name + ": ")); err != nil {
			return err
		}
	}

	return nil
}

func (o *historyObserver) OnToken(ctx context.Context, token string) error {
	if _, err := o.w.Write([]byte(token)); err != nil {
		return err
	}

	return nil
}

func (o *historyObserver) OnEnd(ctx context.Context) error {
	if _, err := o.w.Write([]byte("\n")); err != nil {
		return err
	}

	return nil
}

func (o *historyObserver) OnError(ctx context.Context, err error) error {
	return nil
}

func (o *historyObserver) OnImage(ctx context.Context, image string) error {
	return nil
}

func (o *historyObserver) OnToolCall(ctx context.Context, toolName string, args any) error {
	return nil
}

func (o *historyObserver) OnToolCallEnd(ctx context.Context, toolName string, result any) error {
	if _, err := o.w.Write(fmt.Appendf(nil, "%s: %v\n", toolName, result)); err != nil {
		return err
	}

	return nil
}
