package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
)

const API_CHAT = "/api/chat"

type Ollama struct {
	baseUrl    string
	httpClient *http.Client
	logger     *slog.Logger
}

func NewOllama(baseUrl string) Llm {
	return &Ollama{
		baseUrl:    baseUrl,
		httpClient: http.DefaultClient,
		logger:     slog.Default(),
	}
}

func (o *Ollama) Chat(ctx context.Context, request ChatRequest) (response ChatResponse, err error) {
	o.logger.Debug("Ollama.Chat called", "model", request.Model, "messages", len(request.Messages), "tools", len(request.Tools), "stream", request.Stream)

	data, err := json.Marshal(request)
	if err != nil {
		return
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseUrl+API_CHAT, bytes.NewBuffer(data))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := o.httpClient.Do(req)
	if err != nil {
		return
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return
	}

	err = json.Unmarshal(resBody, &response)

	o.logger.Debug("Ollama.Chat response", "status", res.StatusCode, "response", string(resBody))

	return response, err
}
