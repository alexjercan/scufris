package pretty

// TODO: implement callback for this

import (
	"fmt"
	"strings"
)

const ANSI_CYAN = "\033[34m"
const ANSI_GRAY = "\033[90m"
const ANSI_RED = "\033[31m"
const ANSI_RESET = "\033[0m"

func PrintName(name string) (int, error) {
	return fmt.Printf("%s%s%s", ANSI_CYAN, name, ANSI_RESET)
}

func PrintToken(token string) (int, error) {
	return fmt.Printf("%s%s%s", ANSI_GRAY, token, ANSI_RESET)
}

func PrintError(err error) (int, error) {
	return fmt.Printf("%s%s%s: %s%s%s\n",
		ANSI_RED, "Error", ANSI_RESET,
		ANSI_GRAY, err.Error(), ANSI_RESET)
}

func PrintImage(image string) (int, error) {
	totalSize := len(image)
	chunkSize := 4096
	steps := totalSize / chunkSize

	buffer := strings.Builder{}

	buffer.WriteString("\033_Gm=1,a=T,f=100;")
	for i := range steps {
		chunk := image[i*chunkSize : (i+1)*chunkSize]

		m := 1
		if i > 0 {
			buffer.WriteString(fmt.Sprintf("\033_Gm=%d;", m))
		}

		buffer.WriteString(chunk)
		buffer.WriteString("\033\\")
	}

	chunk := image[steps*chunkSize:]
	buffer.WriteString("\033_Gm=0;")
	buffer.WriteString(chunk)
	buffer.WriteString("\033\\")

	return fmt.Print(buffer.String())
}
