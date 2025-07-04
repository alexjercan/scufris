package llm

import (
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

func (o *Ollama) Chat(ctx context.Context, request ChatRequest) (response ChatResponse, err error) {
	data, err := json.Marshal(request)
	if err != nil {
		return
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

	res, err := o.httpClient.Do(req)
	if err != nil {
		return response, &scufris.Error{
			Code:    "OLLAMA_REQUEST_ERROR",
			Message: "failed to make Ollama request",
			Err:     fmt.Errorf("failed to make Ollama request: %w", err),
		}
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return response, &scufris.Error{
			Code:    "OLLAMA_RESPONSE_ERROR",
			Message: "failed to read Ollama response",
			Err:     fmt.Errorf("failed to read Ollama response: %w", err),
		}
	}

	if res.StatusCode != http.StatusOK {
		return response, &scufris.Error{
			Code:    "OLLAMA_RESPONSE_ERROR",
			Message: fmt.Sprintf("Ollama request failed with status code %d", res.StatusCode),
			Err:     fmt.Errorf("Ollama request failed with status code %d: %s", res.StatusCode, string(resBody)),
		}
	}

	err = json.Unmarshal(resBody, &response)
	if err != nil {
		return response, &scufris.Error{
			Code:    "OLLAMA_RESPONSE_UNMARSHAL_ERROR",
			Message: "failed to unmarshal Ollama response",
			Err:     fmt.Errorf("failed to unmarshal Ollama response: %w", err),
		}
	}

	return response, nil
}
