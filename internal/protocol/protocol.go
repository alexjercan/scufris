package protocol

import (
	"context"
	"encoding/gob"

	"github.com/alexjercan/scufris/tool"
)

type ProtocolCallback struct {
	enc *gob.Encoder
}

func NewProtocolCallback(enc *gob.Encoder) *ProtocolCallback {
	return &ProtocolCallback{
		enc: enc,
	}
}

func (p *ProtocolCallback) OnStart(ctx context.Context, name string) error {
	return p.enc.Encode(NewMessage(MessageOnStart, PayloadOnStart{name}))
}

func (p *ProtocolCallback) OnToken(ctx context.Context, token string) error {
	return p.enc.Encode(NewMessage(MessageOnToken, PayloadOnToken{token}))
}

func (p *ProtocolCallback) OnEnd(ctx context.Context) error {
	return p.enc.Encode(NewMessage(MessageOnEnd, PayloadOnEnd{}))
}

func (p *ProtocolCallback) OnToolCall(ctx context.Context, name string, tool string, params tool.ToolParameters) error {
	return p.enc.Encode(NewMessage(MessageOnToolCall, PayloadOnToolCall{Caller: name, ToolName: tool, Args: params.String()}))
}

func (p *ProtocolCallback) OnToolResponse(ctx context.Context, name string, tool string, response tool.ToolResponse) error {
	return p.enc.Encode(NewMessage(MessageOnToolResponse, PayloadOnToolResponse{Caller: name, ToolName: tool, Result: response.String()}))
}

func (p *ProtocolCallback) OnError(ctx context.Context, err error) error {
	return p.enc.Encode(NewMessage(MessageOnError, PayloadOnError{Err: err.Error()}))
}

func (p *ProtocolCallback) OnImage(ctx context.Context, image string) error {
	return p.enc.Encode(NewMessage(MessageOnImage, PayloadOnImage{Image: image}))
}

func (p *ProtocolCallback) OnResponse(ctx context.Context, input string) error {
	return p.enc.Encode(NewMessage(MessageResponse, PayloadResponse{Response: input}))
}
