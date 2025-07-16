package verbose

import (
	"context"
	"fmt"

	"github.com/alexjercan/scufris"
	"github.com/alexjercan/scufris/internal/contextkeys"
	"github.com/alexjercan/scufris/internal/observer"
	"github.com/alexjercan/scufris/internal/registry"
)

const ANSI_CYAN = "\033[34m"
const ANSI_GRAY = "\033[90m"
const ANSI_RED = "\033[31m"
const ANSI_RESET = "\033[0m"

type verboseObserver struct {
}

func NewVerboseObserver() observer.Observer {
	return &verboseObserver{}
}

func (v *verboseObserver) OnStart(ctx context.Context) {
	if name, ok := contextkeys.AgentName(ctx); ok {
		fmt.Printf("%s%s%s: ", ANSI_CYAN, name, ANSI_RESET)
	}
}

func (v *verboseObserver) OnToken(ctx context.Context, token string) error {
	fmt.Printf("%s%s%s", ANSI_GRAY, token, ANSI_RESET)

	return nil
}

func (v *verboseObserver) OnEnd(ctx context.Context) {
	fmt.Println()
}

func (v *verboseObserver) OnError(ctx context.Context, err error) {
	if err, ok := err.(*scufris.Error); ok {
		fmt.Printf("%s%s%s: [%s%s%s] %s%s%s\n",
			ANSI_RED, "Error", ANSI_RESET,
			ANSI_CYAN, err.Code, ANSI_RESET,
			ANSI_GRAY, err.Message, ANSI_RESET)
	} else {
		fmt.Printf("%s%s%s: %s%s%s\n",
			ANSI_RED, "Error", ANSI_RESET,
			ANSI_GRAY, err.Error(), ANSI_RESET)
	}
}

func (v *verboseObserver) OnImage(ctx context.Context, imageId string) error {
	if img, ok := registry.GetImage(ctx, imageId); ok {
		iCat(img)
	} else {
		return &scufris.Error{
			Code:    "IMAGE_NOT_FOUND",
			Message: fmt.Sprintf("image with id %s not found in registry", imageId),
			Err:     fmt.Errorf("image with id %s not found in registry", imageId),
		}
	}

	return nil
}

func (v *verboseObserver) OnToolCall(ctx context.Context, toolName string, args any) error {
	return nil
}

func (v *verboseObserver) OnToolCallEnd(ctx context.Context, toolName string, result any) error {
	fmt.Printf("%s%s%s: %s%s%s\n", ANSI_CYAN, toolName, ANSI_RESET, ANSI_GRAY, result, ANSI_RESET)

	return nil
}

func iCat(image string) {
	totalSize := len(image)
	chunkSize := 4096
	steps := totalSize / chunkSize

	fmt.Printf("\033_Gm=1,a=T,f=100;")
	for i := range steps {
		chunk := image[i*chunkSize : (i+1)*chunkSize]

		m := 1
		if i > 0 {
			fmt.Printf("\033_Gm=%d;", m)
		}

		fmt.Print(chunk)
		fmt.Printf("\033\\")
	}

	chunk := image[steps*chunkSize:]
	fmt.Printf("\033_Gm=%d;", 0)
	fmt.Print(chunk)
	fmt.Printf("\033\\")

	fmt.Println()
}
