package socket

import (
	"context"
	"encoding/gob"

	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/observer"
)

type socketObserver struct {
	enc *gob.Encoder
}

func NewSocketObserver(enc *gob.Encoder) observer.Observer {
	return &socketObserver{
		enc: enc,
	}
}

func (o *socketObserver) OnStart(ctx context.Context) error {
	if name, ok := contextkeys.AgentName(ctx); ok {
		return o.enc.Encode(NewMessage(MessageOnStart, PayloadOnStart{name}))
	}

	return nil
}

func (o *socketObserver) OnToken(ctx context.Context, token string) error {
	return o.enc.Encode(NewMessage(MessageOnToken, PayloadOnToken{token}))
}

func (o *socketObserver) OnEnd(ctx context.Context) error {
	return o.enc.Encode(NewMessage(MessageOnEnd, PayloadOnEnd{}))
}

func (o *socketObserver) OnError(ctx context.Context, err error) error {
	return o.enc.Encode(NewMessage(MessageOnError, PayloadOnError{err.Error()}))
}

func (o *socketObserver) OnImage(ctx context.Context, image string) error {
	return o.enc.Encode(NewMessage(MessageOnImage, PayloadOnImage{image}))
}

func (o *socketObserver) OnToolCall(ctx context.Context, toolName string, args any) error {
	return o.enc.Encode(NewMessage(MessageOnToolCall, PayloadOnToolCall{toolName, args}))
}

func (o *socketObserver) OnToolCallEnd(ctx context.Context, toolName string, result any) error {
	return o.enc.Encode(NewMessage(MessageOnToolCallEnd, PayloadOnToolCallEnd{toolName, result}))
}
