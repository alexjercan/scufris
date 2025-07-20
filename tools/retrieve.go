package tools

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/knowledge"
	"github.com/alexjercan/scufris/internal/observer"
)

type RetrieveToolParameters struct {
	Query string `json:"query" jsonschema:"title=query,description=The thing we want to retrieve from the knowledge base"`
}

func (p *RetrieveToolParameters) Validate(tool Tool) error {
	if p.Query == "" {
		return fmt.Errorf("query cannot be empty")
	}
	return nil
}

type RetrieveTool struct {
	maxResults int

	retriever *knowledge.Retriever
	logger    *slog.Logger
}

func NewRetrieveTool(maxResults int, retriever *knowledge.Retriever) Tool {
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

func (t *RetrieveTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	t.logger.Debug("RetrieveTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	query := params.(*RetrieveToolParameters).Query

	observer.OnStart(ctx)
	err := observer.OnToken(ctx, fmt.Sprintf("I need to retrieve information for: %s", query))
	if err != nil {
		return nil, err
	}
	observer.OnEnd(ctx)

	results, err := t.retriever.Retrieve(ctx, knowledge.NewRetrieverRequest(query, t.maxResults))
	if err != nil {
		return nil, &scufris.Error{
			Code:    "RETRIEVER_ERROR",
			Message: "failed to retrieve information from knowledge base",
			Err:     fmt.Errorf("failed to retrieve information from knowledge base: %w", err),
		}
	}

	search := ""
	for _, result := range results {
		search = search + fmt.Sprintf("Content: %s\n", result.Content)
	}

	err = observer.OnToolCallEnd(ctx, t.Name(), search)
	if err != nil {
		return nil, err
	}

	t.logger.Debug("RetrieveTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("query", query),
		slog.Int("results_count", len(results)),
	)

	return search, nil
}
