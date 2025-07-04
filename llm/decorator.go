package llm

import "log/slog"

type LlmWrapper struct {
	llm Llm
}

func NewLlmWrapper(llm Llm) *LlmWrapper {
	return &LlmWrapper{
		llm: llm,
	}
}

func (w *LlmWrapper) WithLogging(logger *slog.Logger) *LlmWrapper {
	w.llm = &loggingLlm{
		llm:    w.llm,
		logger: logger,
	}
	return w
}

func (w *LlmWrapper) Build() Llm {
	return w.llm
}
