package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"

	"github.com/invopop/jsonschema"
	orderedmap "github.com/wk8/go-ordered-map/v2"
)

const OLLAMA_URL = "http://localhost:11434"
const MODEL_NAME = "llama3.2:1b"

type MessageRole int

const (
	RoleSystem MessageRole = iota
	RoleUser
	RoleAssistant
	RoleTool
)

var messageRole = map[MessageRole]string{
	RoleSystem:    "system",
	RoleUser:      "user",
	RoleAssistant: "assistant",
	RoleTool:      "tool",
}

type FunctionToolCall struct {
	Name string `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type ToolCall struct {
	Function FunctionToolCall `json:"function"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

func NewMessage(role MessageRole, content string) Message {
	return Message{
		messageRole[role],
		content,
		[]ToolCall {},
	}
}

type FunctionTool struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Parameters  jsonschema.Schema `json:"parameters"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function FunctionTool `json:"function"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools"`
	Stream   bool      `json:"stream"`
}

type ChatResponse struct {
	Message Message `json:"message"`
}

type Llm struct {
	model string
}

func NewLlm(model string) Llm {
	return Llm{model}
}

func (this Llm) Chat(messages []Message, tools []Tool) (message Message, err error) {
	path, err := url.JoinPath(OLLAMA_URL, "/api/chat")
	if err != nil {
		return
	}

	request := ChatRequest{this.model, messages, tools, false}
	data, err := json.Marshal(request)
	if err != nil {
		return
	}

	res, err := http.Post(path, "application/json", bytes.NewBuffer(data))
	if err != nil {
		return
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return
	}

	response := ChatResponse{}
	err = json.Unmarshal(resBody, &response)

	return response.Message, err
}

func main() {
	llm := NewLlm(MODEL_NAME)
	history := []Message{}
	tools := []Tool{}

	properties := orderedmap.New[string, *jsonschema.Schema]()
	properties.AddPairs(
		orderedmap.Pair[string, *jsonschema.Schema]{
			Key: "lhs",
			Value: &jsonschema.Schema{
				Type:        "number",
				Description: "The left hand side of the addition",
			},
		},
		orderedmap.Pair[string, *jsonschema.Schema]{
			Key: "rhs",
			Value: &jsonschema.Schema{
				Type:        "number",
				Description: "The right hand side of the addition",
			},
		})

	tools = append(tools, Tool{
		"function",
		FunctionTool{
			"calculator",
			"Use this tool to do additions on numbers",
			jsonschema.Schema{
				Type:       "object",
				Properties: properties,
				Required:   []string{"lhs", "rhs"},
			},
		},
	})

	for {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(err)
			return
		}
		history = append(history, NewMessage(RoleUser, prompt))
		message, err := llm.Chat(history, tools)
		if err != nil {
			fmt.Println(err)
			return
		}

		if len(message.ToolCalls) != 0 {
			history = append(history, message)

			for _, call := range message.ToolCalls {
				if call.Function.Name == "calculator" {
					lhs, _ := strconv.Atoi(call.Function.Arguments["lhs"].(string))
					rhs, _ := strconv.Atoi(call.Function.Arguments["rhs"].(string))
					result := lhs + rhs

					history = append(history, NewMessage(RoleTool, strconv.Itoa(result)))
					message, err = llm.Chat(history, tools)
					if err != nil {
						fmt.Println(err)
						return
					}
				}
			}
		}

		fmt.Println(message.Content)
		history = append(history, message)
	}
}
