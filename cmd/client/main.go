package main

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"log/slog"
	"net"
	"os"

	"github.com/alexjercan/scufris/internal/logging"
	"github.com/alexjercan/scufris/internal/pretty"
	"github.com/alexjercan/scufris/internal/socket"
)

const ANSI_CYAN = "\033[34m"
const ANSI_GRAY = "\033[90m"
const ANSI_RED = "\033[31m"
const ANSI_RESET = "\033[0m"

func main() {
	socket.MessageInit()

	logging.SetupLogger(slog.LevelInfo, "text")

	c, err := net.Dial("unix", socket.SOCKET_PATH)
	if err != nil {
		panic(err)
	}
	defer c.Close()

	enc := gob.NewEncoder(c)
	dec := gob.NewDecoder(c)

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		prompt, err := reader.ReadString('\n')
		if err != nil {
			panic(err)
		}

		if prompt == "" {
			return
		}

		err = enc.Encode(socket.NewMessage(socket.MessagePrompt, socket.PayloadPrompt{Prompt: prompt}))
		if err != nil {
			panic(err)
		}

	LOOP:
		for {
			var m socket.Message
			if err := dec.Decode(&m); err != nil {
				panic(err)
			}

			switch m.Kind {

			case socket.MessageResponse:
				response := m.Payload.(socket.PayloadResponse).Response
				fmt.Println(response)
				break LOOP

			case socket.MessageOnEnd:
				pretty.OnEnd()
			case socket.MessageOnError:
				err := m.Payload.(socket.PayloadOnError).Err
				pretty.OnError(err)
			case socket.MessageOnImage:
				img := m.Payload.(socket.PayloadOnImage).Image
				pretty.OnImage(img)
			case socket.MessageOnStart:
				name := m.Payload.(socket.PayloadOnStart).Name
				pretty.OnStart(name)
			case socket.MessageOnToken:
				token := m.Payload.(socket.PayloadOnToken).Token
				pretty.OnToken(token)
			case socket.MessageOnToolCall:
				toolCall := m.Payload.(socket.PayloadOnToolCall)
				pretty.OnToolCall(toolCall.ToolName, toolCall.Args)
			case socket.MessageOnToolCallEnd:
				toolCall := m.Payload.(socket.PayloadOnToolCallEnd)
				pretty.OnToolCallEnd(toolCall.ToolName, toolCall.Result)
			default:
				panic(fmt.Sprintf("unexpected socket.MessageKind: %#v", m.Kind))
			}
		}
	}
}
