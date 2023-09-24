# Scufris Service

This example implements a client server approach for Scufris. This is best used
in cases where you want to start scufris when you boot (e.g with systemd) and
then be able to create a session with the client when you need to start it.

### Quickstart

Start the server and then in another terminal start the client.

```console
python server.py
python client.py
```
