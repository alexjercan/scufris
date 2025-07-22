package tools

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/websearch"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type WebSearchToolParameters struct {
	Query string `json:"query" jsonschema:"title=query,description=The thing we want to search on the web"`
}

func (p *WebSearchToolParameters) Validate(tool tool.Tool) error {
	if p.Query == "" {
		return fmt.Errorf("query cannot be empty")
	}
	return nil
}

func (p *WebSearchToolParameters) String() string {
	return fmt.Sprintf("query: %s", p.Query)
}

type WebSearchToolResponse struct {
	Search string `json:"search" jsonschema:"title=search,description=The search results from the web"`
}

func (r *WebSearchToolResponse) String() string {
	return fmt.Sprintf("%s", r.Search)
}

func (r *WebSearchToolResponse) Image() uuid.UUID {
	return uuid.Nil
}

type WebSearchTool struct {
	maxResults int

	client websearch.WebSearchClient
	logger *slog.Logger
}

func NewWebSearchTool(maxResults int) tool.Tool {
	return &WebSearchTool{
		maxResults: maxResults,
		client:     websearch.NewDdgClient(),
		logger:     slog.Default(),
	}
}

func (t *WebSearchTool) Parameters() reflect.Type {
	return reflect.TypeOf(WebSearchToolParameters{})
}

func (t *WebSearchTool) Name() string {
	return "websearch"
}

func (t *WebSearchTool) Description() string {
	return "Use this tool to search the web for additional information; IMPORTANT the query MUST be a valid string"
}

func (t *WebSearchTool) Call(ctx context.Context, params tool.ToolParameters) (tool.ToolResponse, error) {
	t.logger.Debug("WebSearchTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	query := params.(*WebSearchToolParameters).Query

	results, err := t.client.Search(ctx, query, t.maxResults)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "WEB_SEARCH_REQUEST_ERROR",
			Message: "failed to make web search request",
			Err:     fmt.Errorf("failed to make web search request: %w", err),
		}
	}

	search := ""
	for _, result := range results {
		search = search + fmt.Sprintf("Title: %s\nInfo: %s\nURL: %s\n", result.Title, result.Info, result.URL)
	}

	t.logger.Debug("WebSearchTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("query", query),
		slog.Int("results_count", len(results)),
	)

	return &WebSearchToolResponse{
		Search: search,
	}, nil
}
