import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import socketserver
import threading
import logging
import io

import torch

logger = logging.getLogger(__name__)

class CheckpointServer:
    def __init__(self, state_dict) -> None:
        self._checkpoint_lock = threading.Lock()
        self._disallowed = False

        ckpt_server = self
        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                """Handle GET requests."""
                self.send_response(200)
                self.send_header('Content-type', 'tensor') # TODO: correct mime type
                self.end_headers()

                # TODO: check step for safety
                
                with ckpt_server._checkpoint_lock:
                    sd = state_dict()

                    torch.save(sd, self.wfile)

        server_address = ('', 0)
        self._server = HTTPServer(server_address, RequestHandler)
        logger.info(f"Started CheckpointServer on {self.address()}...")

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
        )
        self._thread.start()

    @classmethod
    def load_from_address(cls, address: str) -> object:
        with urllib.request.urlopen(address) as f:
            data = f.read()

        reader = io.BytesIO(data)
        return torch.load(reader)

    def address(self) -> str:
        port = self._server.socket.getsockname()[1]
        return f"http://{socket.gethostname()}:{port}"


    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.exception("got exception in checkpoint server")

    def disallow_checkpoint(self) -> None:
        if not self._disallowed:
            self._disallowed = True
            self._checkpoint_lock.acquire()

    def allow_checkpoint(self) -> None:
        if self._disallowed:
            self._disallowed = False
            self._checkpoint_lock.release()