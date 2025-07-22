package history

import (
	"context"
	"fmt"
	"io"

	"github.com/alexjercan/scufris/tool"
)

type HistoryCallback struct {
	w io.Writer
}

func NewHistoryCallback(w TranscriptWriter) *HistoryCallback {
	return &HistoryCallback{
		w: w,
	}
}

func (h *HistoryCallback) OnStart(ctx context.Context, name string) error {
	_, err := fmt.Fprintf(h.w, "%s: ", name)
	return err
}

func (h *HistoryCallback) OnToken(ctx context.Context, token string) error {
	_, err := fmt.Fprint(h.w, token)
	return err
}

func (h *HistoryCallback) OnEnd(ctx context.Context) error {
	_, err := fmt.Fprintln(h.w)
	return err
}

func (h *HistoryCallback) OnToolCall(ctx context.Context, name string, tool string, params tool.ToolParameters) error {
	s := params.String()
	if s == "" {
		_, err := fmt.Fprintf(h.w, "%s: I will call the %s tool.\n", name, tool)
		return err
	} else {
		_, err := fmt.Fprintf(h.w, "%s: I will call the %s tool with parameters: %s\n", name, tool, s)
		return err
	}
}

func (h *HistoryCallback) OnToolResponse(ctx context.Context, name string, tool string, response tool.ToolResponse) error {
	_, err := fmt.Fprintf(h.w, "%s: The %s tool returned: %s\n", name, tool, response.String())
	return err
}

func (h *HistoryCallback) OnPrompt(ctx context.Context, input string) error {
	_, err := fmt.Fprintf(h.w, "User: %s\n", input)
	return err
}

func (h *HistoryCallback) OnError(ctx context.Context, err error) error {
	_, err = fmt.Fprintf(h.w, "Error: %v\n", err)
	return err
}
