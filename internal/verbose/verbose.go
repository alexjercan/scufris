package verbose

import "fmt"

const ANSI_CYAN = "\033[34m"
const ANSI_GRAY = "\033[90m"
const ANSI_RESET = "\033[0m"

func Say(name string, message string) {
	fmt.Printf("%s%s%s: %s%s%s\n", ANSI_CYAN, name, ANSI_RESET, ANSI_GRAY, message, ANSI_RESET)
}
