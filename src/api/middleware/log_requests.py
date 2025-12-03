import logging
import json
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

logger = logging.getLogger("deep_request_logger")
logging.basicConfig(level=logging.INFO)

print("### SAFE DEEP LOGGING MIDDLEWARE LOADED ###", __file__)


class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        # -----------------------------------------------------------
        # Correlation ID for pairing request/response
        # -----------------------------------------------------------
        rid = str(uuid.uuid4())[:8]

        # Read request details
        method = request.method
        path = request.url.path
        query = request.url.query
        client = request.client.host if request.client else "unknown"

        # -----------------------------------------------------------
        # Read request body without breaking downstream
        # -----------------------------------------------------------
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            body_bytes = b""
            body_text = "<unreadable>"

        try:
            parsed_json = json.loads(body_text)
        except Exception:
            parsed_json = None

        print("\n─────────────────────────────────────────────")
        print(f"[RID {rid}] >>> REQUEST >>>")
        print(f"Client: {client}")
        print(f"Method: {method}")
        print(f"Path:   {path}")
        print(f"Query:  {query}")
        print("Headers:")
        for k, v in request.headers.items():
            print(f"  {k}: {v}")

        print("Body (raw):", body_text)
        if parsed_json is not None:
            print("Body (json):", json.dumps(parsed_json, indent=2))

        # Timer
        start = time.time()

        # -----------------------------------------------------------
        # Capture response body safely using a wrapper
        # -----------------------------------------------------------
        send_buffer = bytearray()

        async def send_wrapper(message):
            """
            Intercepts ASGI 'http.response.body' messages to capture the body
            BEFORE it reaches Mangum. Works for streaming or normal responses.
            """
            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                send_buffer.extend(body)

            await send(message)

        # Use Starlette's ASGI function signature
        async def app(scope, receive, send):
            return await request.app(scope, receive, send_wrapper)

        # Execute downstream app
        await app(request.scope, request.receive, lambda m: None)

        # Now call_next normally (real response)
        response = await call_next(request)

        # -----------------------------------------------------------
        # Print response AFTER call_next
        # -----------------------------------------------------------
        duration_ms = round((time.time() - start) * 1000, 2)

        resp_bytes = bytes(send_buffer)
        resp_text = resp_bytes.decode("utf-8", errors="replace")
        try:
            resp_json = json.loads(resp_text)
        except Exception:
            resp_json = None

        print(f"[RID {rid}] <<< RESPONSE <<<")
        print("Status:", response.status_code)
        print("Time:", duration_ms, "ms")
        print("Response Body (raw):", resp_text)
        if resp_json is not None:
            print("Response Body (json):", json.dumps(resp_json, indent=2))
        print("─────────────────────────────────────────────\n")

        # DO NOT MODIFY response.body_iterator — return response untouched
        return response
