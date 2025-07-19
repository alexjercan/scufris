package history

import (
	"bytes"
	"fmt"
	"log/slog"
	"os"

	"github.com/alexjercan/scufris"
)

type FileTranscriptWriter struct {
	buffer   bytes.Buffer
	filename string
	logger   *slog.Logger
}

func NewFileTranscriptWriter(filename string) TranscriptSink {
	return &FileTranscriptWriter{
		filename: filename,
		logger:   slog.Default(),
	}
}

func (t *FileTranscriptWriter) Write(p []byte) (int, error) {
	return t.buffer.Write(p)
}

func (t *FileTranscriptWriter) Close() error {
	file, err := os.Create(t.filename)
	if err != nil {
		return &scufris.Error{
			Code:    "TRANSCRIPT_WRITE_ERROR",
			Message: "failed to create transcript file",
			Err:     fmt.Errorf("failed to create transcript file: %w", err),
		}
	}
	defer file.Close()

	if _, err := file.Write(t.buffer.Bytes()); err != nil {
		return &scufris.Error{
			Code:    "TRANSCRIPT_WRITE_ERROR",
			Message: "failed to write transcript to file",
			Err:     fmt.Errorf("failed to write transcript to file: %w", err),
		}
	}

	t.logger.Debug("Transcript written to file", slog.String("filename", t.filename))
	return nil
}
