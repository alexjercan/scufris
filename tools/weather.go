package tools

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/observer"
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
	logger     *slog.Logger
}

func NewWeatherTool() Tool {
	return &WeatherTool{
		baseUrl:    "https://wttr.in/",
		httpClient: http.DefaultClient,
		logger:     slog.Default(),
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
	t.logger.Debug("WeatherTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	city := params.(*WeatherToolParameters).City

	observer.OnStart(ctx)
	err := observer.OnToken(ctx, fmt.Sprintf("I need to check the weather in: %s", city))
	if err != nil {
		return nil, err
	}
	observer.OnEnd(ctx)

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

	if res.StatusCode != http.StatusOK {
		return nil, &scufris.Error{
			Code:    "WEATHER_RESPONSE_ERROR",
			Message: "unexpected weather response status",
			Err:     fmt.Errorf("unexpected weather response status: %s", res.Status),
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
	err = observer.OnToolCallEnd(ctx, t.Name(), response)
	if err != nil {
		return nil, err
	}

	t.logger.Debug("WeatherTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("city", city),
		slog.String("response", response),
	)

	return map[string]string{
		"city":    city,
		"weather": response,
	}, nil
}
