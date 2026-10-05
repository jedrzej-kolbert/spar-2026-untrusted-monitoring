"""inspect-ai's openai provider must work with the locked openai SDK.

Serves one chat completion from a local HTTP server (no API call) and requests it
through get_model("openai/..."), as the APPS/BigCodeBench generators and monitors do.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        reply = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "<score>1</score>"},
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


async def test_openai_provider_round_trip(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    try:
        model = get_model(
            "openai/gpt-4o-mini",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            api_key="test",
        )
        out = await model.generate(
            [ChatMessageUser(content="hi")], config=GenerateConfig(max_retries=0)
        )
    finally:
        server.shutdown()
    assert out.completion == "<score>1</score>"
