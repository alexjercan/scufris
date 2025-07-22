package tools

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type WeatherToolParameters struct {
	City string `json:"city" jsonschema:"title=city,description=The city for which to get the weather"`
}

func (p *WeatherToolParameters) Validate(tool tool.Tool) error {
	if p.City == "" {
		return fmt.Errorf("city cannot be empty")
	}
	return nil
}

func (p *WeatherToolParameters) String() string {
	return fmt.Sprintf("city: %s", p.City)
}

type WeatherToolResponse struct {
	Weather string `json:"weather" jsonschema:"title=weather,description=The weather information for the specified city"`
}

func (r *WeatherToolResponse) String() string {
	return fmt.Sprintf("%s", r.Weather)
}

func (r *WeatherToolResponse) Image() uuid.UUID {
	return uuid.Nil
}

type WeatherTool struct {
	baseUrl    string
	httpClient *http.Client
	logger     *slog.Logger
}

func NewWeatherTool() tool.Tool {
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

func (t *WeatherTool) Call(ctx context.Context, params tool.ToolParameters) (tool.ToolResponse, error) {
	t.logger.Debug("WeatherTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	city := params.(*WeatherToolParameters).City

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

	t.logger.Debug("WeatherTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("city", city),
		slog.String("response", response),
	)

	return &WeatherToolResponse{
		Weather: response,
	}, nil
}
