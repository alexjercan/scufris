package tools

import (
	"context"
	"fmt"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/verbose"
	"github.com/alexjercan/scufris/internal/websearch"
)

type WebSearchToolParameters struct {
	Query string `json:"query" jsonschema:"title=query,description=The thing we want to search on the web"`
}

func (p *WebSearchToolParameters) Validate() error {
	if p.Query == "" {
		return fmt.Errorf("query cannot be empty")
	}
	return nil
}

type WebSearchTool struct {
	maxResults int

	client websearch.WebSearchClient
}

func NewWebSearchTool(maxResults int) Tool {
	return &WebSearchTool{
		maxResults: maxResults,
		client:     websearch.NewDdgClient(),
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

func (t *WebSearchTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	query := params.(*WebSearchToolParameters).Query

	if name, ok := contextkeys.AgentName(ctx); ok {
		verbose.Say(name, fmt.Sprintf("I need to search the web for: %s", query))
	}

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

	verbose.Say("websearch", search)

	return map[string]any{
		"query":   query,
		"results": search,
	}, nil
}
