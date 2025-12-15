# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: API REQUEST / RESPONSE LOGGING MIDDLEWARE
# REQUIREMENTS SATISFIED: backend observability and debugging support

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/api/middleware/log_requests.py

Defines a custom ASGI middleware for detailed HTTP request and response logging.

This middleware intercepts incoming HTTP requests and outgoing responses
to capture full request metadata, bodies, response payloads, status codes,
and end-to-end latency. Each request is tagged with a unique request ID
to make tracing across logs easier during local development and AWS
deployment.

Key features:
    - Logs HTTP method and request path
    - Captures and prints request and response bodies
    - Attempts structured JSON pretty-printing when possible
    - Measures per-request latency in milliseconds
    - Safely passes through non-HTTP ASGI events

This middleware is intended for debugging, testing, and operational
visibility during Phase 2 deployment. It is not user-facing and does not
modify request or response behavior.
"""
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

