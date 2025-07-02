package tools

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
)


type WeatherToolParameters struct {
	City string `json:"city" jsonschema:"title=city,description=The city for which to get the weather"`
}

func (p *WeatherToolParameters) Validate() error {
	if p.City == "" {
		return fmt.Errorf("city cannot be empty")
	}
	return nil
}

type WeatherTool struct {
	Params WeatherToolParameters

	baseUrl    string
	httpClient *http.Client
	logger	 *slog.Logger
}

func NewWeatherTool() Tool {
	return &WeatherTool{
		baseUrl:    "https://wttr.in/",
		httpClient: http.DefaultClient,
		logger:     slog.Default(),
	}
}

func (t *WeatherTool) Name() string {
	return "weather"
}

func (t *WeatherTool) Description() string {
	return "Use this tool to get the weather for a specific city; IMPORTANT: the city MUST be a valid string"
}

func (t *WeatherTool) Call(ctx context.Context) (any, error) {
	t.logger.Debug("WeatherTool.Call called", "city", t.Params.City)

	req, err := http.NewRequestWithContext(ctx, "GET", t.baseUrl+t.Params.City+"?format=3", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "text/plain")

	t.logger.Debug("WeatherTool.Call Sending request to weather API", "url", req.URL.String())

	res, err := t.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get weather: %w", err)
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	response := string(resBody)
	t.logger.Debug("WeatherTool.Call received response", "response", response)

	return map[string]string{
		"city":    t.Params.City,
		"weather": response,
	}, nil
}

