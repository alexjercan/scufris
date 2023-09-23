import socket
import threading
import logging
from rich.logging import RichHandler
from .agent import create_agent
from .util import recv_until

rich_handler = RichHandler(rich_tracebacks=True)
logging.root.addHandler(rich_handler)
logging.root.setLevel(logging.ERROR)

logger = logging.getLogger("scufris")
logger.setLevel(logging.INFO)


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


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.bind((HOST, PORT))

    server_socket.listen(5)
    logger.info(f"Listening on {HOST}:{PORT}")

    while True:
        client_socket, address = server_socket.accept()
        logger.info(f"Accepted connection from {address}")

        threading.Thread(target=handle_client, args=(client_socket,)).start()
