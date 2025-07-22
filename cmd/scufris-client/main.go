package main

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"

	"github.com/alexjercan/scufris/internal/config"
	"github.com/alexjercan/scufris/internal/pretty"
	"github.com/alexjercan/scufris/internal/protocol"
)

const ANSI_CYAN = "\033[34m"
const ANSI_GRAY = "\033[90m"
const ANSI_RED = "\033[31m"
const ANSI_RESET = "\033[0m"

func main() {
	config.SetupLogger(slog.LevelInfo, "text") // Setup Logger
	protocol.MessageInit()                     // Initialize the protocol messages
	cfg := config.MustLoadClientConfig()       // Load the client configuration

	// Dial the server socket
	c, err := net.Dial("unix", cfg.SocketPath)
	if err != nil {
		panic(err)
	}
	defer c.Close()

	// Create encoders and decoders for the connection
	enc := gob.NewEncoder(c)
	dec := gob.NewDecoder(c)

	// Very basic client loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				fmt.Println("Exiting...")
				return
			}

			panic(fmt.Sprintf("failed to read input: %v", err))
		}

		if prompt == "" {
			return
		}

		err = enc.Encode(protocol.NewMessage(protocol.MessagePrompt, protocol.PayloadPrompt{Prompt: prompt}))
		if err != nil {
			panic(err)
		}

	LOOP:
		for {
			var m protocol.Message
			if err := dec.Decode(&m); err != nil {
				if err == io.EOF {
					fmt.Println("Server closed the connection")
					return
				}

				panic(err)
			}

			switch m.Kind {

			case protocol.MessageResponse:
				response := m.Payload.(protocol.PayloadResponse).Response
				fmt.Println(response)
				break LOOP

			case protocol.MessageOnEnd:
				fmt.Println()
			case protocol.MessageOnError:
				err := m.Payload.(protocol.PayloadOnError).Err
				pretty.PrintError(fmt.Errorf("%s", err))
			case protocol.MessageOnImage:
				img := m.Payload.(protocol.PayloadOnImage).Image
				pretty.PrintImage(img)
				fmt.Println()
			case protocol.MessageOnStart:
				name := m.Payload.(protocol.PayloadOnStart).Name
				pretty.PrintName(name)
				fmt.Printf(": ")
			case protocol.MessageOnToken:
				token := m.Payload.(protocol.PayloadOnToken).Token
				pretty.PrintToken(token)
			case protocol.MessageOnToolCall:
				toolCall := m.Payload.(protocol.PayloadOnToolCall)
				pretty.PrintName(toolCall.Caller)
				fmt.Printf(": ")
				if toolCall.Args == "" {
					pretty.PrintToken(fmt.Sprintf("I will call the %s tool.", toolCall.ToolName))
				} else {
					pretty.PrintToken(fmt.Sprintf("I will call the %s tool with parameters: %s", toolCall.ToolName, toolCall.Args))
				}
				fmt.Println()
			case protocol.MessageOnToolResponse:
				toolCall := m.Payload.(protocol.PayloadOnToolResponse)
				pretty.PrintName(toolCall.Caller)
				fmt.Printf(": ")
				pretty.PrintToken(fmt.Sprintf("The %s tool returned: %s", toolCall.ToolName, toolCall.Result))
				fmt.Println()
			default:
				panic(fmt.Sprintf("unexpected protocol.MessageKind: %#v", m.Kind))
			}
		}
	}
}
