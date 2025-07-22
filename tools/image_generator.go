package tools

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/registry"
	"github.com/alexjercan/scufris/tool"
	"github.com/google/uuid"
)

type ImageGeneratorToolParameters struct {
	Prompt string `json:"prompt" jsonschema:"title=prompt,description=The text prompt to generate an image from."`
}

func (p *ImageGeneratorToolParameters) Validate(tool tool.Tool) error {
	if p.Prompt == "" {
		return fmt.Errorf("prompt cannot be empty")
	}
	return nil
}

func (p *ImageGeneratorToolParameters) String() string {
	return fmt.Sprintf("prompt: %s", p.Prompt)
}

type ImageGeneratorToolResponse struct {
	ImageId uuid.UUID `json:"image_id" jsonschema:"title=image_id,description=The ID of the generated image."`
}

func (r *ImageGeneratorToolResponse) String() string {
	return fmt.Sprintf("image_id: %s", r.ImageId)
}

func (r *ImageGeneratorToolResponse) Image() uuid.UUID {
	return r.ImageId
}

type ImageGeneratorTool struct {
	gen    imagegen.ImageGenerator
	registry registry.ImageRegistry

	logger *slog.Logger
}

func NewImageGeneratorTool(gen imagegen.ImageGenerator, registry registry.ImageRegistry) tool.Tool {
	return &ImageGeneratorTool{
		gen:    gen,
		registry: registry,
		logger: slog.Default(),
	}
}

func (t *ImageGeneratorTool) Parameters() reflect.Type {
	return reflect.TypeOf(ImageGeneratorToolParameters{})
}

func (t *ImageGeneratorTool) Name() string {
	return "image-generator"
}

func (t *ImageGeneratorTool) Description() string {
	return "Use this tool to generate an image using a text prompt; IMPORTANT: the prompt MUST be a valid string"
}

func (t *ImageGeneratorTool) Call(ctx context.Context, params tool.ToolParameters) (tool.ToolResponse, error) {
	t.logger.Debug("ImageGeneratorTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	prompt := params.(*ImageGeneratorToolParameters).Prompt

	data, err := t.gen.Generate(ctx, imagegen.NewGenerateRequest(prompt))
	if err != nil {
		return nil, err
	}

	img := base64.StdEncoding.EncodeToString(data)
	imageId, err := t.registry.AddImage(ctx, img)
	if err != nil {
		return nil, err
	}

	t.logger.Debug("ImageGeneratorTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("image_id", imageId.String()),
	)

	return &ImageGeneratorToolResponse{
		ImageId: imageId,
	}, nil
}
