import socket
from rich.prompt import Prompt
from rich import print as pprint


def recv_until(socket: socket.socket, delimiter: str = '\n', buffer_size: int = 4096) -> str:
    """Receive data from a socket until a specific delimiter is found.

    Parameters
    ----------
    socket : socket.socket
        The socket to read from.
    delimiter : str, optional
        The character to use as a delimiter (default is '\n').
    buffer_size : int, optional
        The size of the buffer used for reading data (default is 4096 bytes).

    Returns
    -------
    str
        The received data as a string, including the delimiter.
    """
    data = b""
    while True:
        chunk = socket.recv(buffer_size)
        if not chunk:
            break
        data += chunk
        if delimiter.encode() in data:
            break

    return data.decode("utf-8")


if __name__ == "__main__":
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
