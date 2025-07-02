package llm

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
