import socket


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
