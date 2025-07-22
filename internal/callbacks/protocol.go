package callbacks

import (
	"context"
	"encoding/gob"

	"github.com/alexjercan/scufris/agent"
	"github.com/alexjercan/scufris/internal/protocol"
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
	return p.enc.Encode(protocol.NewMessage(protocol.MessageOnStart, protocol.PayloadOnStart{Name: name}))
}

func (p *ProtocolCallback) OnToken(ctx context.Context, token string) error {
	return p.enc.Encode(protocol.NewMessage(protocol.MessageOnToken, protocol.PayloadOnToken{Token: token}))
}

func (p *ProtocolCallback) OnEnd(ctx context.Context) error {
	return p.enc.Encode(protocol.NewMessage(protocol.MessageOnEnd, protocol.PayloadOnEnd{}))
}

func (p *ProtocolCallback) OnToolCall(ctx context.Context, name string, tool string, params tool.ToolParameters) error {
	return p.enc.Encode(protocol.NewMessage(protocol.MessageOnToolCall, protocol.PayloadOnToolCall{Caller: name, ToolName: tool, Args: params.String()}))
}

func (p *ProtocolCallback) OnToolResponse(ctx context.Context, name string, tool string, response tool.ToolResponse) error {
	return p.enc.Encode(protocol.NewMessage(protocol.MessageOnToolResponse, protocol.PayloadOnToolResponse{Caller: name, ToolName: tool, Result: response.String()}))
}

func (p *ProtocolCallback) OnError(ctx context.Context, err error) error {
	return p.enc.Encode(protocol.NewMessage(protocol.MessageOnError, protocol.PayloadOnError{Err: err.Error()}))
}

func (p *ProtocolCallback) OnImage(ctx context.Context, image string) error {
	return p.enc.Encode(protocol.NewMessage(protocol.MessageOnImage, protocol.PayloadOnImage{Image: image}))
}

func (p *ProtocolCallback) OnResponse(ctx context.Context, input string) error {
	return p.enc.Encode(protocol.NewMessage(protocol.MessageResponse, protocol.PayloadResponse{Response: input}))
}

func (p *ProtocolCallback) ToCallbacks() agent.CrewCallbacks {
	return agent.CrewCallbacks{
		OnStart:         p.OnStart,
		OnToken:         p.OnToken,
		OnEnd:           p.OnEnd,
		OnToolCall:      p.OnToolCall,
		OnToolResponse:  p.OnToolResponse,
		OnImage:         p.OnImage,
	}
}
