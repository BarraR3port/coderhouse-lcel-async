import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class FakeOpenAI:
    base_url: str = ""
    status: int = 200
    body: dict[str, Any] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def answer(self, content: str) -> None:
        self.status, self.body = 200, completion(content)

    def fail(self, status: int, message: str) -> None:
        self.status, self.body = status, {"error": {"message": message, "type": "error"}}


@contextmanager
def running_fake_openai() -> Iterator[FakeOpenAI]:
    fake = FakeOpenAI()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            fake.requests.append(json.loads(self.rfile.read(length)))
            payload = json.dumps(fake.body).encode()
            self.send_response(fake.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    fake.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()


def completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
    }
