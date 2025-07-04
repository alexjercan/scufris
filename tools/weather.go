package tools

import (
	"context"
	"fmt"
	"io"
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
}

func NewWeatherTool() Tool {
	return &WeatherTool{
		baseUrl:    "https://wttr.in/",
		httpClient: http.DefaultClient,
	}
}

func (t *WeatherTool) Name() string {
	return "weather"
}

func (t *WeatherTool) Description() string {
	return "Use this tool to get the weather for a specific city; IMPORTANT: the city MUST be a valid string"
}

func (t *WeatherTool) Call(ctx context.Context) (any, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", t.baseUrl+t.Params.City+"?format=3", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "text/plain")

	res, err := t.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get weather: %w", err)
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	response := string(resBody)

	return map[string]string{
		"city":    t.Params.City,
		"weather": response,
	}, nil
}
