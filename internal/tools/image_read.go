package tools

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"os"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/registry"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type ImageReadToolParameters struct {
	Path string `json:"path" jsonschema:"title=path,description=The path to use to read the image."`
}

func (p *ImageReadToolParameters) Validate(tool tool.Tool) error {
	if p.Path == "" {
		return fmt.Errorf("path cannot be empty")
	}
	return nil
}

func (p *ImageReadToolParameters) String() string {
	return fmt.Sprintf("path: %s", p.Path)
}

type ImageReadToolResponse struct {
	ImageId uuid.UUID `json:"image_id" jsonschema:"title=image_id,description=The ID of the read image."`
}

func (r *ImageReadToolResponse) String() string {
	return fmt.Sprintf("image_id: %s", r.ImageId)
}

func (r *ImageReadToolResponse) Image() uuid.UUID {
	return r.ImageId
}

type ImageReadTool struct {
	registry registry.Registry

	logger *slog.Logger
}

func NewImageReadTool(registry registry.Registry) tool.Tool {
	return &ImageReadTool{
		registry: registry,
		logger:   slog.Default(),
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

func (t *ImageReadTool) Call(ctx context.Context, params tool.ToolParameters) (tool.ToolResponse, error) {
	t.logger.Debug("ImageReadTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	path := params.(*ImageReadToolParameters).Path

	dat, err := os.ReadFile(path)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "READ_ERROR",
			Message: "failed to read file",
			Err:     fmt.Errorf("failed to read file: %w", err),
		}
	}

	img := base64.StdEncoding.EncodeToString(dat)
	imageId, err := t.registry.AddImage(ctx, img, nil) // TODO: we might want to add options here in the future
	if err != nil {
		return nil, err
	}

	t.logger.Debug("ImageReadTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("image_id", imageId.String()),
	)

	return &ImageReadToolResponse{
		ImageId: imageId,
	}, nil
}
