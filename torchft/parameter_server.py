from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
import uuid
import logging
import urllib.request
import json

from torch.distributed import TCPStore

from torchft.process_group import ProcessGroup


logger: logging.Logger = logging.getLogger(__name__)


class ParameterServer(ABC):
    """
    This implements a threaded parameter server using the torchft reconfigurable
    ProcessGroups.
    """

    def __init__(self, port: int) -> None:
        self.store = TCPStore(
            host_name="0.0.0.0",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

        ps = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/new_session":
                    self.send_response(400)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.err(f"invalid path, got {self.path}")
                    return

                try:
                    self.send_response(200)
                    self.send_header(
                        "Content-type", "application/json"
                    )  # TODO: correct mime type
                    self.end_headers()

                    session_id = str(uuid.uuid4())

                    store_addr = (
                        f"{socket.gethostname()}:{ps.store.port}/session/{session_id}"
                    )

                    logger.info(f"creating new session {session_id}")

                    data = (
                        json.dumps(
                            {
                                "session_id": session_id,
                                "store_addr": store_addr,
                            }
                        )
                        + "\n"
                    )
                    data = data.encode()

                    self.wfile.write(data)

                    # close the connection up front so client will know json is
                    # complete
                    self.finish()
                    self.connection.close()

                    # hijack thread for the session
                    ps.handle_session(session_id, store_addr)
                except Exception:
                    logger.exception(
                        f"got exception in request handler for {self.path}"
                    )
                    raise

        server_address = ("", port)
        self._server = ThreadingHTTPServer(server_address, RequestHandler)
        self._server.daemon_threads = True
        logger.info(f"Started ParameterServer on {self.address()}...")

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def load_from_address(cls, address: str) -> object:
        logger.info(f"fetching checkpoint from {address}")

        reader = io.BytesIO(data)
        return torch.load(reader, weights_only=True)

    def address(self) -> str:
        port = self._server.socket.getsockname()[1]
        return f"http://{socket.gethostname()}:{port}/new_session"

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.exception("got exception in checkpoint server")

    @classmethod
    @abstractmethod
    def new_process_group(cls) -> ProcessGroup: ...

    @classmethod
    def new_session(cls, address: str) -> ProcessGroup:
        with urllib.request.urlopen(address) as f:
            data = json.load(f)

        session_id = data["session_id"]
        store_addr = data["store_addr"]

        logger.info(f"connecting to session {session_id} at {store_addr}")

        pg = cls.new_process_group()
        # client is always rank 1
        pg.configure(store_addr, rank=1, world_size=2)

        return pg

    def handle_session(self, session_id: str, store_addr: str) -> None:
        pg = self.new_process_group()
        # paramter server is always rank 0
        pg.configure(store_addr, rank=0, world_size=2)

        self.forward(session_id, pg)

    @abstractmethod
    def forward(self, session_id: str, pg: ProcessGroup) -> None: ...
