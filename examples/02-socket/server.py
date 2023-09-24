import socket
import threading
import logging
from rich.logging import RichHandler
from scufris import create_agent

rich_handler = RichHandler(rich_tracebacks=True)
logging.root.addHandler(rich_handler)
logging.root.setLevel(logging.ERROR)

logger = logging.getLogger("scufris")
logger.setLevel(logging.INFO)


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


HOST = "0.0.0.0"
PORT = 42069


def handle_client(client_socket):
    agent = create_agent()

    while True:
        try:
            prompt = recv_until(client_socket).strip()
            if not prompt:
                break

            logger.debug(f"Received prompt: '{prompt}'")

            response = agent.run(prompt)
            if not response.endswith("\n"):
                response += "\n"

            client_socket.sendall(response.encode("utf-8"))
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception(e)
            break

    logger.info(f"Closing connection to {client_socket.getpeername()}")
    client_socket.close()


if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.bind((HOST, PORT))

    server_socket.listen(5)
    logger.info(f"Listening on {HOST}:{PORT}")

    threads = []
    while True:
        try:
            client_socket, address = server_socket.accept()
            logger.info(f"Accepted connection from {address}")

            task = threading.Thread(target=handle_client, args=(client_socket,))
            task.start()

            threads.append(task)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception(e)

    logger.info("Shutting down...")
    for task in threads:
        task.join()
