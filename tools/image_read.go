package tools

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"os"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
)

type ImageReadToolParameters struct {
	Path string `json:"path" jsonschema:"title=path,description=The path to use to read the image."`
}

func (p *ImageReadToolParameters) Validate() error {
	if p.Path == "" {
		return fmt.Errorf("path cannot be empty")
	}
	return nil
}

type ImageReadTool struct {
	logger *slog.Logger
}

func NewImageReadTool() Tool {
	return &ImageReadTool{
		logger: slog.Default(),
	}
}

func (t *ImageReadTool) Parameters() reflect.Type {
	return reflect.TypeOf(ImageReadToolParameters{})
}

func (t *ImageReadTool) Name() string {
	return "image-read"
}

func (t *ImageReadTool) Description() string {
	return "Use this tool to read image data from a path; this tool will return the image id; IMPORTANT: the path MUST be a valid string"
}

func (t *ImageReadTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	t.logger.Debug("ImageReadTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	path := params.(*ImageReadToolParameters).Path

	observer.OnStart(ctx)
	err := observer.OnToken(ctx, fmt.Sprintf("I need to read the image from path: %s", path))
	if err != nil {
		return nil, err
	}
	observer.OnEnd(ctx)

	dat, err := os.ReadFile(path)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "READ_ERROR",
			Message: "failed to read file",
			Err:     fmt.Errorf("failed to read file: %w", err),
		}
	}

	img := base64.StdEncoding.EncodeToString(dat)
	imageId, err := registry.AddImage(ctx, img)
	if err != nil {
		return nil, err
	}

	err = observer.OnImage(ctx, img)
	if err != nil {
		return nil, err
	}

	t.logger.Debug("ImageReadTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("image_id", imageId),
	)

	return map[string]string{
		"image_id": imageId,
	}, nil
}
