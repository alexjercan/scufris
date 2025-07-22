package main

import (
	"context"
	"fmt"

	"github.com/alexjercan/scufris/llm"
)

func main() {
	ctx := context.Background()
	l := llm.NewOllama("http://localhost:11434")

	request := llm.NewChatRequest(
		"qwen3:latest",
		[]llm.Message{
			llm.NewMessage(llm.RoleUser, "Hello, how are you?"),
		},
		[]llm.ToolInfo{},
		true,
	)
	response, err := l.Chat(ctx, request, &llm.ChatOptions{
		OnStart: func(ctx context.Context) error {
			fmt.Println("Chat started")
			return nil
		},
		OnEnd: func(ctx context.Context) error {
			fmt.Println("Chat ended")
			return nil
		},
		OnToken: func(ctx context.Context, token string) error {
			fmt.Printf("Got token: %s\n", token)
			return nil
		},
	})

	if err != nil {
		panic(fmt.Errorf("failed to chat: %w", err))
	}

	fmt.Println("Response:", response.Message.Content)
}
