package llm

import "context"

type ChatRequest struct {
	Model    string     `json:"model"`
	Messages []Message  `json:"messages"`
	Tools    []ToolInfo `json:"tools"`
	Stream   bool       `json:"stream"`
}

func NewChatRequest(model string, messages []Message, tools []ToolInfo, stream bool) ChatRequest {
	return ChatRequest{
		Model:    model,
		Messages: messages,
		Tools:    tools,
		Stream:   stream,
	}
}

type ChatResponse struct {
	Message Message `json:"message"`
}

type EmbeddingsRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

func NewEmbeddingsRequest(model string, input string) EmbeddingsRequest {
	return EmbeddingsRequest{
		Model: model,
		Input: input,
	}
}

type EmbeddingsResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

const (
	ModelInfoVision     = "vision"
	ModelInfoCompletion = "completion"
)

type ModelInfoResponse struct {
	// TODO: Add more fields as needed
	Capabilities []string `json:"capabilities"`
}

type ChatOptions struct {
	OnStart func(ctx context.Context) error
	OnEnd   func(ctx context.Context) error
	OnToken func(ctx context.Context, token string) error
}
