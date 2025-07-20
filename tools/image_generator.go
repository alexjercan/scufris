package tools

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"reflect"

	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
)

type ImageGeneratorToolParameters struct {
	Prompt string `json:"prompt" jsonschema:"title=prompt,description=The text prompt to generate an image from."`
}

func (p *ImageGeneratorToolParameters) Validate() error {
	if p.Prompt == "" {
		return fmt.Errorf("prompt cannot be empty")
	}
	return nil
}

type ImageGeneratorTool struct {
	gen imagegen.ImageGenerator
	logger *slog.Logger
}

func NewImageGeneratorTool(gen imagegen.ImageGenerator) Tool {
	return &ImageGeneratorTool{
		gen: gen,
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

func (t *ImageGeneratorTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	t.logger.Debug("ImageGeneratorTool.Call called",
		slog.String("name", t.Name()),
		slog.Any("params", params),
	)

	prompt := params.(*ImageGeneratorToolParameters).Prompt

	observer.OnStart(ctx)
	err := observer.OnToken(ctx, fmt.Sprintf("I need to generate an image with the prompt: %s", prompt))
	if err != nil {
		return nil, err
	}
		observer.OnEnd(ctx)

	data, err := t.gen.Generate(ctx, imagegen.NewGenerateRequest(prompt))
	if err != nil {
		return nil, err
	}

	img := base64.StdEncoding.EncodeToString(data)
	imageId, err := registry.AddImage(ctx, img)
	if err != nil {
		return nil, err
	}

	err = observer.OnImage(ctx, img)
	if err != nil {
		return nil, err
	}

	t.logger.Debug("ImageGeneratorTool.Call completed",
		slog.String("name", t.Name()),
		slog.String("image_id", imageId),
	)

	return map[string]string{
		"image_id": imageId,
	}, nil
}
