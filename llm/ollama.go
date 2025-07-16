package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/observer"
)

const API_CHAT = "/api/chat"

type Ollama struct {
	baseUrl    string
	httpClient *http.Client
	logger *slog.Logger
}

func NewOllama(baseUrl string) Llm {
	return &Ollama{
		baseUrl:    baseUrl,
		httpClient: http.DefaultClient,
		logger:     slog.Default(),
	}
}

func (o *Ollama) Chat(ctx context.Context, request ChatRequest) (response ChatResponse, err error) {
	o.logger.Debug("Ollama.Chat called",
		slog.String("model", request.Model),
		slog.Int("messages", len(request.Messages)),
		slog.Int("tools", len(request.Tools)),
		slog.Bool("stream", request.Stream),
	)

	observer.OnStart(ctx)
	defer observer.OnEnd(ctx)

	data, err := json.Marshal(request)
	if err != nil {
		return response, &scufris.Error{
			Code:    "OLLAMA_REQUEST_MARSHAL_ERROR",
			Message: "failed to marshal Ollama request",
			Err:     fmt.Errorf("failed to marshal Ollama request: %w", err),
		}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseUrl+API_CHAT, bytes.NewBuffer(data))
	if err != nil {
		return response, &scufris.Error{
			Code:    "OLLAMA_REQUEST_ERROR",
			Message: "failed to create Ollama request",
			Err:     fmt.Errorf("failed to create Ollama request: %w", err),
		}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/x-ndjson")

	res, err := o.httpClient.Do(req)
	if err != nil {
		return response, &scufris.Error{
			Code:    "OLLAMA_REQUEST_ERROR",
			Message: "failed to make Ollama request",
			Err:     fmt.Errorf("failed to make Ollama request: %w", err),
		}
	}

	if res.StatusCode != http.StatusOK {
		resBody, _ := io.ReadAll(res.Body)

		return response, &scufris.Error{
			Code:    "OLLAMA_RESPONSE_ERROR",
			Message: fmt.Sprintf("Ollama request failed with status code %d", res.StatusCode),
			Err:     fmt.Errorf("Ollama request failed with status code %d: %s", res.StatusCode, string(resBody)),
		}
	}

	scanner := bufio.NewScanner(res.Body)
	response.Message = NewMessage(RoleAssistant, "")

	for scanner.Scan() {
		bts := scanner.Bytes()

		var token ChatResponse
		err = json.Unmarshal(bts, &token)
		if err != nil {
			return response, &scufris.Error{
				Code:    "OLLAMA_RESPONSE_UNMARSHAL_ERROR",
				Message: "failed to unmarshal Ollama response",
				Err:     fmt.Errorf("failed to unmarshal Ollama response: %w", err),
			}
		}

		err = observer.OnToken(ctx, token.Message.Content)
		if err != nil {
			return response, err
		}

		response.Message = response.Message.Append(token.Message)
	}

	o.logger.Debug("Ollama.Chat response",
		slog.String("content", response.Message.Content),
		slog.Any("tool_calls", response.Message.ToolCalls),
	)

	return response, nil
}
