package tools

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"reflect"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/verbose"
)

type ReadImageToolParameters struct {
	Path string `json:"path" jsonschema:"title=path,description=The path to use to read the image."`
}

func (p *ReadImageToolParameters) Validate() error {
	if p.Path == "" {
		// TODO: maybe validate if path exists
		return fmt.Errorf("path cannot be empty")
	}
	return nil
}

type ReadImageTool struct {
}

func NewReadImageTool() Tool {
	return &ReadImageTool{}
}

func (t *ReadImageTool) Parameters() reflect.Type {
	return reflect.TypeOf(ReadImageToolParameters{})
}

func (t *ReadImageTool) Name() string {
	return "read-image"
}

func (t *ReadImageTool) Description() string {
	return "Use this tool to read image data from a path; this tool will return the image id; IMPORTANT: the path MUST be a valid string"
}

func (t *ReadImageTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	path := params.(*ReadImageToolParameters).Path

	if name, ok := contextkeys.AgentName(ctx); ok {
		verbose.Say(name, fmt.Sprintf("I need to read the image %s", path))
	}

	dat, err := os.ReadFile(path)
	if err != nil {
		return nil, &scufris.Error{
			Code:    "READ_ERROR",
			Message: "failed to read file",
			Err:     fmt.Errorf("failed to read file: %w", err),
		}
	}

	img := base64.StdEncoding.EncodeToString(dat)
	imageId := DefaultImageRegistry.AddImage(img)

	verbose.ICat(img)

	return map[string]string{
		"status":   "Success",
		"image_id": imageId,
	}, nil
}
