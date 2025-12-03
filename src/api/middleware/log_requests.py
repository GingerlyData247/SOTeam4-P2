import json
import uuid
import time
from typing import Callable, Awaitable
from starlette.types import ASGIApp, Receive, Scope, Send


print("### ULTRA-STABLE ASGI LOGGER LOADED ###")


class DeepASGILogger:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = str(uuid.uuid4())[:8]
        method = scope.get("method")
        path = scope.get("path")

        # ------------------------------
        # Capture request body
        # ------------------------------
        body_bytes = b""

        async def recv_wrapper():
            nonlocal body_bytes
            msg = await receive()
            if msg["type"] == "http.request":
                body_bytes += msg.get("body", b"")
            return msg

        # ------------------------------
        # Prepare response capture
        # ------------------------------
        resp_body = b""
        status_code = None

        async def send_wrapper(message):
            nonlocal resp_body, status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

            if message["type"] == "http.response.body":
                resp_body += message.get("body", b"")

            await send(message)

        start = time.time()

        # ------------------------------
        # Call main app
        # ------------------------------
        await self.app(scope, recv_wrapper, send_wrapper)

        duration_ms = round((time.time() - start) * 1000, 2)

        # ------------------------------
        # Logging
        # ------------------------------
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[RID {rid}] >>> REQUEST >>>")
        print(f"Method: {method}")
        print(f"Path:   {path}")

        raw_body = body_bytes.decode("utf-8", errors="replace")
        print("Request Body:", raw_body)

        try:
            print("Request JSON:", json.dumps(json.loads(raw_body), indent=2))
        except:
            pass

        print(f"[RID {rid}] <<< RESPONSE <<<")
        print("Status:", status_code)
        print("Time:", duration_ms, "ms")

        raw_resp = resp_body.decode("utf-8", errors="replace")
        print("Response Body (raw):", raw_resp)
        try:
            print("Response JSON:", json.dumps(json.loads(raw_resp), indent=2))
        except:
            pass
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
