import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import socketserver
import threading
import logging
import io

import torch

logger: logging.Logger = logging.getLogger(__name__)


class CheckpointServer:
    def __init__(self, state_dict) -> None:
        self._checkpoint_lock = threading.Lock()
        self._disallowed = False
        self._step = -1

        ckpt_server = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                with ckpt_server._checkpoint_lock:
                    step = ckpt_server._step

                    if self.path != f"/checkpoint/{step}":
                        self.send_response(400)
                        self.send_header("Content-type", "text/plain")
                        self.end_headers()
                        self.err(
                            f"invalid checkpoint requested, serving {step} but got {self.path}"
                        )
                        return

                    self.send_response(200)
                    self.send_header(
                        "Content-type", "tensor"
                    )  # TODO: correct mime type
                    self.end_headers()

                    sd = state_dict()

                    torch.save(sd, self.wfile)

            def err(self, msg: str) -> None:
                logger.error(msg)
                self.wfile.write(msg.encode())

        server_address = ("", 0)
        self._server = HTTPServer(server_address, RequestHandler)
        logger.info(f"Started CheckpointServer on {self.address()}...")

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def load_from_address(cls, address: str) -> object:
        logger.info(f"fetching checkpoint from {address}")

        with urllib.request.urlopen(address) as f:
            data = f.read()

        reader = io.BytesIO(data)
        return torch.load(reader, weights_only=True)

    def address(self) -> str:
        port = self._server.socket.getsockname()[1]
        return f"http://{socket.gethostname()}:{port}/checkpoint/{self._step}"

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.exception("got exception in checkpoint server")

    def disallow_checkpoint(self) -> None:
        if not self._disallowed:
            self._disallowed = True
            self._checkpoint_lock.acquire()

    def allow_checkpoint(self, step: int) -> None:
        self._step = step

        if self._disallowed:
            self._disallowed = False
            self._checkpoint_lock.release()

    def shutdown(self) -> None:
        self._server.shutdown()
