package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const API_CHAT = "/api/chat"

const OLLAMA_ERROR = "OLLAMA_ERROR"

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

	if res.StatusCode != http.StatusOK {
		return response, &LlmError{
			Code:    OLLAMA_ERROR,
			Message: fmt.Sprintf("failed to make request %v", request),
			Err:     fmt.Errorf("status code %d: %s", res.StatusCode, string(resBody)),
		}
	}

	err = json.Unmarshal(resBody, &response)

	return response, err
}
