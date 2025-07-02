# scufris

Scufris = Scuffed Jarvis

**scufris** is a modular Go framework for building AI assistants powered by LLMs and tool-calling.
It supports plug-and-play LLM providers like [Ollama](https://ollama.com) and custom tools for dynamic responses.

### Quickstart

```console
go run ./cmd/scufris
```

This launches a simple REPL interface where you can type messages and interact with an LLM:

```
> What's the weather in Tokyo?
Tokyo's current weather is hot with a temperature of +26°C.
```

Make sure Ollama is running and a model like llama3.2:1b is available:

```console
ollama serve
ollama pull llama3.2:1b
```

### Chatting with the Agent

You can also use the `scufris` package to create your own agent programmatically:

```go
func main() {
	ctx := context.Background()

	client := llm.NewOllama(OLLAMA_URL)
	agent := scufris.NewAgent("llama3.2:1b", client)

	response, err := agent.Chat(ctx, "What's the weather in Tokyo?")
	fmt.Println(response)
}
```

### Adding a Tool

To add a custom tool, create a new Go file in the `tools` directory. For example, to add a adder tool:

```go
package tools

type AdderToolParameters struct {
    Lhs int `json:"lhs" jsonschema:"title=lhs,description=The left-hand side operand"`
    Rhs int `json:"rhs" jsonschema:"title=rhs,description=The right-hand side operand"`
}

func (p *AdderToolParameters) Validate() error {
	return nil // Add validation logic if needed
}

type AdderTool struct {
	Params AdderToolParameters

	logger	 *slog.Logger
}

func NewAdderTool() Tool {
	return &AdderTool{
		logger: slog.Default(),
	}
}

func (t *AdderTool) Name() string {
    return "adder"
}

func (t *AdderTool) Description() string {
    return "Adds two numbers together"
}

func (t *AdderTool) Call(ctx context.Context) (any, error) {
    result := t.Params.Lhs + t.Params.Rhs
    t.logger.Debug("AdderTool called", "result", result)

    return map[string]any{"result": result}, nil
}
```

Then, you just have to add the tool to the `agent`

```go
func main() {
    agent := scufris.NewAgent(...)

    agent.AddFunctionTool(tools.NewAdderTool())
}
```
