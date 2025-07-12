package imagegen

import "log/slog"

type ImageGeneratorWrapper struct {
	gen ImageGenerator
}

func NewImageGeneratorWrapper(gen ImageGenerator) *ImageGeneratorWrapper {
	return &ImageGeneratorWrapper{
		gen: gen,
	}
}

func (w *ImageGeneratorWrapper) WithLogging(logger *slog.Logger) *ImageGeneratorWrapper {
	w.gen = &loggingImageGenerator{
		gen:    w.gen,
		logger: logger,
	}
	return w
}

func (w *ImageGeneratorWrapper) Build() ImageGenerator {
	return w.gen
}
