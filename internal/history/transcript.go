package history

import (
	"log/slog"
	"strings"

	"github.com/alexjercan/scufris/registry"
)

type TranscriptWriter interface {
	Options() registry.TextOptions
	Write(p []byte) (int, error)
	String() string
}

type FileTranscriptWriter struct {
	strings.Builder
	filename string
	logger   *slog.Logger
}

func NewFileTranscriptWriter(filename string) TranscriptWriter {
	return &FileTranscriptWriter{
		filename: filename,
		logger:   slog.Default(),
	}
}

func (t *FileTranscriptWriter) Options() registry.TextOptions {
	return &registry.MapTextOptions{Path: t.filename}
}
