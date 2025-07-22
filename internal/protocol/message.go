package protocol

import (
	"encoding/gob"
)

type MessageKind int

const (
	MessagePrompt MessageKind = iota
	MessageResponse

	MessageOnStart
	MessageOnToken
	MessageOnEnd
	MessageOnError
	MessageOnImage
	MessageOnToolCall
	MessageOnToolResponse
)

type Message struct {
	Kind    MessageKind
	Payload any
}

func NewMessage(kind MessageKind, payload any) Message {
	return Message{
		Kind:    kind,
		Payload: payload,
	}
}

type PayloadPrompt struct {
	Prompt string
}

type PayloadResponse struct {
	Response string
}

type PayloadOnStart struct {
	Name string
}

type PayloadOnToken struct {
	Token string
}

type PayloadOnEnd struct{}

type PayloadOnError struct {
	Err string
}

type PayloadOnImage struct {
	Image string
}

type PayloadOnToolCall struct {
	Caller   string
	ToolName string
	Args     string
}

type PayloadOnToolResponse struct {
	Caller   string
	ToolName string
	Result   string
}

func MessageInit() {
	gob.Register(PayloadPrompt{})
	gob.Register(PayloadResponse{})

	gob.Register(PayloadOnStart{})
	gob.Register(PayloadOnToken{})
	gob.Register(PayloadOnEnd{})

	gob.Register(PayloadOnError{})

	gob.Register(PayloadOnImage{})

	gob.Register(PayloadOnToolCall{})
	gob.Register(PayloadOnToolResponse{})
}
