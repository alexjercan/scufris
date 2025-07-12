package tools

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/verbose"
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
	baseUrl    string
	httpClient *http.Client
}

func NewWeatherTool() Tool {
	return &WeatherTool{
		baseUrl:    "https://wttr.in/",
		httpClient: http.DefaultClient,
	}
}

func (t *WeatherTool) Parameters() reflect.Type {
	return reflect.TypeOf(WeatherToolParameters{})
}

func (t *WeatherTool) Name() string {
	return "weather"
}

func (t *WeatherTool) Description() string {
	return "Use this tool to get the weather for a specific city; IMPORTANT: the city MUST be a valid string"
}

func (t *WeatherTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	city := params.(*WeatherToolParameters).City

	if name, ok := contextkeys.AgentName(ctx); ok {
		verbose.Say(name, fmt.Sprintf("I need to check the weather in: %s", city))
	}

	req, err := http.NewRequestWithContext(ctx, "GET", t.baseUrl+city+"?format=3", nil)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "WEATHER_REQUEST_ERROR",
			Message: "failed to create weather request",
			Err:     fmt.Errorf("failed to create weather request: %w", err),
		}
	}
	req.Header.Set("Accept", "text/plain")

	res, err := t.httpClient.Do(req)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "WEATHER_REQUEST_ERROR",
			Message: "failed to make weather request",
			Err:     fmt.Errorf("failed to make weather request: %w", err),
		}
	}

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "WEATHER_RESPONSE_ERROR",
			Message: "failed to read weather response",
			Err:     fmt.Errorf("failed to read weather response: %w", err),
		}
	}

	response := string(resBody)
	verbose.Say("wttr.in", response)

	return map[string]string{
		"city":    city,
		"weather": response,
	}, nil
}
