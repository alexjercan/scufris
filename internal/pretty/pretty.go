package pretty

import (
	"fmt"

	"github.com/alexjercan/scufris"
)

const ANSI_CYAN = "\033[34m"
const ANSI_GRAY = "\033[90m"
const ANSI_RED = "\033[31m"
const ANSI_RESET = "\033[0m"

func OnStart(name string) error {
	fmt.Printf("%s%s%s: ", ANSI_CYAN, name, ANSI_RESET)

	return nil
}

func OnToken(token string) error {
	fmt.Printf("%s%s%s", ANSI_GRAY, token, ANSI_RESET)

	return nil
}

func OnEnd() error {
	fmt.Println()

	return nil
}

func OnError(err error) error {
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

	return nil
}

func OnImage(img string) error {
	iCat(img)

	return nil
}

func OnToolCall(toolName string, args any) error {
	return nil
}

func OnToolCallEnd(toolName string, result any) error {
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
