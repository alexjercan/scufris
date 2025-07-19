package history

import "io"

type TranscriptSink interface {
	io.WriteCloser
}
