import socket
from rich.prompt import Prompt
from rich import print as pprint
from .util import recv_until


def main():
    host = "127.0.0.1"
    port = 42069

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    while True:
        prompt = Prompt.ask("User")
        if not prompt:
            break
        if not prompt.endswith("\n"):
            prompt += "\n"

        client.send(prompt.encode("utf-8"))

        response = recv_until(client).strip()
        pprint(f"[bold blue]Scufris:[/bold blue] {response}")

    client.close()
