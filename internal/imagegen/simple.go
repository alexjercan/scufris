package imagegen

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
)

const API_GENERATE = "/generate"

type Simple struct {
	baseUrl    string
	httpClient *http.Client
	logger     *slog.Logger
}

func NewSimple(baseUrl string) ImageGenerator {
	return &Simple{
		baseUrl:    baseUrl,
		httpClient: http.DefaultClient,
		logger:     slog.Default(),
	}
}

func (g *Simple) Generate(ctx context.Context, request GenerateRequest) (img []byte, err error) {
	g.logger.Info("Generating image", "prompt", request.Prompt)

	data, err := json.Marshal(request)
	if err != nil {
		return img, &Error{
			Code:    "IMAGEGEN_ERROR",
			Message: "failed to marshal request",
			Err:     fmt.Errorf("failed to marshal request: %w", err),
		}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", g.baseUrl+API_GENERATE, bytes.NewBuffer(data))
	if err != nil {
		return img, &Error{
			Code:    "IMAGEGEN_ERROR",
			Message: "failed to create image generate request",
			Err:     fmt.Errorf("failed to create image generate request: %w", err),
		}
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := g.httpClient.Do(req)
	if err != nil {
		return img, &Error{
			Code:    "IMAGEGEN_ERROR",
			Message: "failed to make image generate request",
			Err:     fmt.Errorf("failed to make image generate request: %w", err),
		}
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return img, &Error{
			Code:    "IMAGEGEN_ERROR",
			Message: "failed to read image generate response",
			Err:     fmt.Errorf("failed to read image generate response: %w", err),
		}
	}

	if res.StatusCode != http.StatusOK {
		return img, &Error{
			Code:    "IMAGEGEN_ERROR",
			Message: fmt.Sprintf("Image Generate request failed with status code %d", res.StatusCode),
			Err:     fmt.Errorf("Image Generate request failed with status code %d: %s", res.StatusCode, string(resBody)),
		}
	}

	g.logger.Debug("Image generated successfully")

	return resBody, nil
}
