package tools

import (
	"context"
	"encoding/base64"
	"fmt"
	"reflect"

	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/imagegen"
	"github.com/alexjercan/scufris/internal/verbose"
)

type ImageToolParameters struct {
	Prompt string `json:"prompt" jsonschema:"title=prompt,description=The text prompt to generate an image from."`
}

func (p *ImageToolParameters) Validate() error {
	if p.Prompt == "" {
		return fmt.Errorf("prompt cannot be empty")
	}
	return nil
}

type ImageTool struct {
	gen imagegen.ImageGenerator
}

func NewImageTool(gen imagegen.ImageGenerator) Tool {
	return &ImageTool{
		gen: gen,
	}
}

func (t *ImageTool) Parameters() reflect.Type {
	return reflect.TypeOf(ImageToolParameters{})
}

func (t *ImageTool) Name() string {
	return "image-generation"
}

func (t *ImageTool) Description() string {
	return "Use this tool to generate an image using a text prompt; IMPORTANT: the prompt MUST be a valid string"
}

func (t *ImageTool) Call(ctx context.Context, params ToolParameters) (any, error) {
	prompt := params.(*ImageToolParameters).Prompt

	if name, ok := contextkeys.AgentName(ctx); ok {
		verbose.Say(name, fmt.Sprintf("I need to run the image generation tool with: %s", prompt))
	}

	data, err := t.gen.Generate(ctx, imagegen.NewGenerateRequest(prompt))
	if err != nil {
		return nil, err
	}

	img := base64.StdEncoding.EncodeToString(data)
	imageId := DefaultImageRegistry.AddImage(img)

	verbose.Say("image-generation", "DONE")
	verbose.ICat(img)

	return map[string]string{
		"status":   "Success",
		"image_id": imageId,
	}, nil
}
