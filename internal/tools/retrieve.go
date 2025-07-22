package tools

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris/registry"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type RetrieveToolParameters struct {
	Query string `json:"query" jsonschema:"title=query,description=The thing we want to retrieve from the knowledge base"`
}

func (p *RetrieveToolParameters) Validate(tool tool.Tool) error {
	if p.Query == "" {
		return fmt.Errorf("query cannot be empty")
	}
	return nil
}

func (p *RetrieveToolParameters) String() string {
	return fmt.Sprintf("query: %s", p.Query)
}

type RetrieveToolResponse struct {
	Search string `json:"search" jsonschema:"title=search,description=The search results from the knowledge base"`
}

func (r *RetrieveToolResponse) String() string {
	return fmt.Sprintf("%s", r.Search)
}

func (r *RetrieveToolResponse) Image() uuid.UUID {
	return uuid.Nil
}

type RetrieveTool struct {
	maxResults int

	retriever registry.Registry
	logger    *slog.Logger
}

func NewRetrieveTool(maxResults int, retriever registry.Registry) tool.Tool {
	return &RetrieveTool{
		maxResults: maxResults,
		retriever:  retriever,
		logger:     slog.Default(),
	}
}

func (t *RetrieveTool) Parameters() reflect.Type {
	return reflect.TypeOf(RetrieveToolParameters{})
}

func (t *RetrieveTool) Name() string {
	return "retrieve"
}

func (t *RetrieveTool) Description() string {
	return "Use this tool to retrieve information from the knowledge base; IMPORTANT the query MUST be a valid string"
}

func (t *RetrieveTool) Call(ctx context.Context, params tool.ToolParameters) (tool.ToolResponse, error) {
	t.logger.Debug("RetrieveTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	query := params.(*RetrieveToolParameters).Query

	ids, err := t.retriever.SearchText(ctx, query, t.maxResults)
	if err != nil {
		return nil, err
	}

	results := make([]string, 0, len(ids))
	for _, id := range ids {
		text, err := t.retriever.GetText(ctx, id)
		if err != nil {
			t.logger.Error("Failed to retrieve text",
				slog.String("id", id.String()),
				slog.String("error", err.Error()),
			)
			continue
		}
		results = append(results, text)
	}

	if len(results) == 0 {
		t.logger.Debug("No results found for query",
			slog.String("name", t.Name()),
			slog.String("query", query),
		)
		return &RetrieveToolResponse{
			Search: "No results found",
		}, nil
	}

	search := ""
	for _, result := range results {
		search = search + fmt.Sprintf("Content: %s\n", result)
	}

	t.logger.Debug("RetrieveTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("query", query),
		slog.Int("results_count", len(results)),
	)

	return &RetrieveToolResponse{
		Search: search,
	}, nil
}
