package llm

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

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

func NewMessage(role MessageRole, content string) Message {
	return Message{
		messageRole[role],
		content,
		[]ToolCall{},
	}
}

func (m Message)Append(o Message) Message {
	m.Content = m.Content + o.Content
	m.ToolCalls = append(m.ToolCalls, o.ToolCalls...)

	return m
}

type FunctionToolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type ToolCall struct {
	Function FunctionToolCall `json:"function"`
}
