package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/alexjercan/scufris"
)

const API_CHAT = "/api/chat"

type Ollama struct {
	baseUrl    string
	httpClient *http.Client
}

func NewOllama(baseUrl string) Llm {
	return &Ollama{
		baseUrl:    baseUrl,
		httpClient: http.DefaultClient,
	}
}

func (o *Ollama) Chat(ctx context.Context, request ChatRequest, onToken ChatOnToken) (response ChatResponse, err error) {
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

		if onToken != nil {
			err = onToken(token.Message.Content)
			if err != nil {
				return response, &scufris.Error{
					Code:    "OLLAMA_TOKEN_CALLBACK_ERROR",
					Message: "failed to call on token Ollama",
					Err:     fmt.Errorf("failed to call on token Ollama: %w", err),
				}
			}
		}
		response.Message = response.Message.Append(token.Message)
	}

	return response, nil
}
