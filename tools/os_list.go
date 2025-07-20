package tools

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/observer"
)

type OsListToolParameters struct {
	Path string `json:"path" jsonschema:"title=path,description=The path to use to list the contents."`
}

func (p *OsListToolParameters) Validate(tool Tool) error {
	if p.Path == "" {
		return fmt.Errorf("path cannot be empty")
	}
	return nil
}

type OsListTool struct {
	logger *slog.Logger
}

func NewOsListTool() Tool {
	return &OsListTool{
		logger: slog.Default(),
	}
}

func (t *OsListTool) Parameters() reflect.Type {
	return reflect.TypeOf(OsListToolParameters{})
}

func (t *OsListTool) Name() string {
	return "os-list"
}

func (t *OsListTool) Description() string {
	return "Use this tool to list files from a path; this tool will return the list of files and directories; IMPORTANT: the path MUST be a valid string; IMPORTANT: this tools does not support ~ expansion, so you must provide the full path."
}

func (t *OsListTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	t.logger.Debug("OsListTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	path := params.(*OsListToolParameters).Path

	observer.OnStart(ctx)
	err := observer.OnToken(ctx, fmt.Sprintf("I need to list the contents of the path: %s", path))
	if err != nil {
		return nil, err
	}
	observer.OnEnd(ctx)

	files, err := os.ReadDir(path)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "OS_LIST_ERROR",
			Message: fmt.Sprintf("failed to list contents of path %s", path),
			Err:     fmt.Errorf("failed to list contents of path %s: %w", path, err),
		}
	}

	var fileList []string
	for _, file := range files {
		fileList = append(fileList, file.Name())
	}

	t.logger.Debug("OsListTool.Call completed",
		slog.String("name", t.Name()),
		slog.Any("fileList", fileList),
	)

	return fmt.Sprintf("%v", fileList), nil
}
