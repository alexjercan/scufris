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
)

const API_CHAT = "/api/chat"
const API_EMBED = "/api/embed"
const API_SHOW = "/api/show"

type Ollama struct {
	baseURL    string
	httpClient *http.Client
	logger     *slog.Logger
}

func NewOllama(baseURL string) Llm {
	return &Ollama{
		baseURL:    baseURL,
		httpClient: http.DefaultClient,
		logger:     slog.Default(),
	}
}

func (o *Ollama) Chat(ctx context.Context, request ChatRequest, opts *ChatOptions) (response ChatResponse, err error) {
	o.logger.Debug("Ollama.Chat called",
		slog.String("model", request.Model),
		slog.Int("messages", len(request.Messages)),
		slog.Int("tools", len(request.Tools)),
		slog.Bool("stream", request.Stream),
	)

	if opts != nil && opts.OnStart != nil {
		if err := opts.OnStart(ctx); err != nil {
			return response, fmt.Errorf("chat start callback error: %w", err)
		}
	}

	defer func() {
		if opts != nil && opts.OnEnd != nil {
			if cbErr := opts.OnEnd(ctx); cbErr != nil && err == nil {
				err = fmt.Errorf("chat end callback error: %w", cbErr)
			}
		}
	}()

	data, err := json.Marshal(request)
	if err != nil {
		return response, &Error{
			Code:    "OLLAMA_REQUEST_MARSHAL_ERROR",
			Message: "failed to marshal Ollama request",
			Err:     fmt.Errorf("failed to marshal Ollama request: %w", err),
		}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+API_CHAT, bytes.NewBuffer(data))
	if err != nil {
		return response, &Error{
			Code:    "OLLAMA_REQUEST_ERROR",
			Message: "failed to create Ollama request",
			Err:     fmt.Errorf("failed to create Ollama request: %w", err),
		}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/x-ndjson")

	res, err := o.httpClient.Do(req)
	if err != nil {
		return response, &Error{
			Code:    "OLLAMA_REQUEST_ERROR",
			Message: "failed to make Ollama request",
			Err:     fmt.Errorf("failed to make Ollama request: %w", err),
		}
	}

	if res.StatusCode != http.StatusOK {
		resBody, _ := io.ReadAll(res.Body)

		return response, &Error{
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
			return response, fmt.Errorf("failed to unmarshal token: %w", err)
		}

		if opts != nil && opts.OnToken != nil {
			if cbErr := opts.OnToken(ctx, token.Message.Content); cbErr != nil {
				return response, fmt.Errorf("token callback error: %w", cbErr)
			}
		}

		response.Message = response.Message.Append(token.Message)
	}

	return response, nil
}

func (o *Ollama) Embeddings(ctx context.Context, request EmbeddingsRequest) (EmbeddingsResponse, error) {
	o.logger.Debug("Ollama.Embeddings called",
		slog.String("model", request.Model),
		slog.Int("input_length", len(request.Input)),
	)

	data, err := json.Marshal(request)
	if err != nil {
		return EmbeddingsResponse{}, &Error{
			Code:    "OLLAMA_EMBEDDINGS_MARSHAL_ERROR",
			Message: "failed to marshal Ollama embeddings request",
			Err:     fmt.Errorf("failed to marshal Ollama embeddings request: %w", err),
		}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+API_EMBED, bytes.NewBuffer(data))
	if err != nil {
		return EmbeddingsResponse{}, &Error{
			Code:    "OLLAMA_EMBEDDINGS_REQUEST_ERROR",
			Message: "failed to create Ollama embeddings request",
			Err:     fmt.Errorf("failed to create Ollama embeddings request: %w", err),
		}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	res, err := o.httpClient.Do(req)
	if err != nil {
		return EmbeddingsResponse{}, &Error{
			Code:    "OLLAMA_EMBEDDINGS_REQUEST_ERROR",
			Message: "failed to make Ollama embeddings request",
			Err:     fmt.Errorf("failed to make Ollama embeddings request: %w", err),
		}
	}

	if res.StatusCode != http.StatusOK {
		resBody, _ := io.ReadAll(res.Body)

		return EmbeddingsResponse{}, &Error{
			Code:    "OLLAMA_EMBEDDINGS_RESPONSE_ERROR",
			Message: fmt.Sprintf("Ollama embeddings request failed with status code %d", res.StatusCode),
			Err:     fmt.Errorf("Ollama embeddings request failed with status code %d: %s", res.StatusCode, string(resBody)),
		}
	}

	var response EmbeddingsResponse
	err = json.NewDecoder(res.Body).Decode(&response)
	if err != nil {
		return EmbeddingsResponse{}, &Error{
			Code:    "OLLAMA_EMBEDDINGS_UNMARSHAL_ERROR",
			Message: "failed to unmarshal Ollama embeddings response",
			Err:     fmt.Errorf("failed to unmarshal Ollama embeddings response: %w", err),
		}
	}

	o.logger.Debug("Ollama.Embeddings response",
		slog.Int("embedding_length", len(response.Embeddings)),
	)

	return response, nil
}

func (o *Ollama) ModelInfo(ctx context.Context, model string) (ModelInfoResponse, error) {
	o.logger.Debug("Ollama.ModelInfo called",
		slog.String("model", model),
	)

	req, err := http.NewRequestWithContext(ctx, "GET", o.baseURL+API_SHOW+"/"+model, nil)
	if err != nil {
		return ModelInfoResponse{}, &Error{
			Code:    "OLLAMA_MODEL_INFO_REQUEST_ERROR",
			Message: "failed to create Ollama model info request",
			Err:     fmt.Errorf("failed to create Ollama model info request: %w", err),
		}
	}
	req.Header.Set("Accept", "application/json")

	res, err := o.httpClient.Do(req)
	if err != nil {
		return ModelInfoResponse{}, &Error{
			Code:    "OLLAMA_MODEL_INFO_REQUEST_ERROR",
			Message: "failed to make Ollama model info request",
			Err:     fmt.Errorf("failed to make Ollama model info request: %w", err),
		}
	}

	if res.StatusCode != http.StatusOK {
		resBody, _ := io.ReadAll(res.Body)

		return ModelInfoResponse{}, &Error{
			Code:    "OLLAMA_MODEL_INFO_RESPONSE_ERROR",
			Message: fmt.Sprintf("Ollama model info request failed with status code %d", res.StatusCode),
			Err:     fmt.Errorf("Ollama model info request failed with status code %d: %s", res.StatusCode, string(resBody)),
		}
	}

	var response ModelInfoResponse
	err = json.NewDecoder(res.Body).Decode(&response)
	if err != nil {
		return ModelInfoResponse{}, &Error{
			Code:    "OLLAMA_MODEL_INFO_UNMARSHAL_ERROR",
			Message: "failed to unmarshal Ollama model info response",
			Err:     fmt.Errorf("failed to unmarshal Ollama model info response: %w", err),
		}
	}

	o.logger.Debug("Ollama.ModelInfo response",
		slog.Any("model_info", response),
	)

	return response, nil
}
