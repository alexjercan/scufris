package imagegen

import (
	"context"
	"log/slog"
)

type loggingImageGenerator struct {
	gen    ImageGenerator
	logger *slog.Logger
}

func (l *loggingImageGenerator) Generate(ctx context.Context, request GenerateRequest) ([]byte, error) {
	l.logger.Info("Generating image", "prompt", request.Prompt)

	image, err := l.gen.Generate(ctx, request)
	if err != nil {
		l.logger.Error("Failed to generate image", "error", err)
		return nil, err
	}

	l.logger.Info("Image generated successfully")
	return image, nil
}
