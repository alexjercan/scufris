package verbose

import "fmt"

const ANSI_CYAN = "\033[34m"
const ANSI_GRAY = "\033[90m"
const ANSI_RESET = "\033[0m"

func Say(name string, message string) {
	fmt.Printf("%s%s%s: %s%s%s\n", ANSI_CYAN, name, ANSI_RESET, ANSI_GRAY, message, ANSI_RESET)
}

func ICat(image string) {
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
