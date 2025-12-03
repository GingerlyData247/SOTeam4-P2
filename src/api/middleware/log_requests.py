import logging
import json
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.requests import Request

logger = logging.getLogger("deep_request_logger")
logging.basicConfig(level=logging.INFO)

print("### DEEP LOGGING MIDDLEWARE LOADED ###", __file__)


class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # -----------------------------------------------------------
        # 1. Correlation ID so request + response pair together
        # -----------------------------------------------------------
        rid = str(uuid.uuid4())[:8]

        method = request.method
        path = request.url.path
        query = request.url.query
        client = request.client.host if request.client else "unknown"

        # -----------------------------------------------------------
        # 2. Read request body without consuming it for downstream
        # -----------------------------------------------------------
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            body_bytes = b""
            body_text = "<unreadable>"

        # Attempt JSON parse
        try:
            parsed_json = json.loads(body_text)
        except Exception:
            parsed_json = None

        # -----------------------------------------------------------
        # 3. Log the incoming request data
        # -----------------------------------------------------------
        print(f"\n──────────────────────────────────────────────────────────")
        print(f"[RID {rid}] >>> REQUEST RECEIVED >>>")
        print(f"Client: {client}")
        print(f"Method: {method}")
        print(f"Path:   {path}")
        print(f"Query:  {query}")

        print(f"Headers:")
        for k, v in request.headers.items():
            print(f"  {k}: {v}")

        print("Body (raw):", body_text)
        if parsed_json is not None:
            print("Body (json):", json.dumps(parsed_json, indent=2))

        # -----------------------------------------------------------
        # 4. Timing
        # -----------------------------------------------------------
        start = time.time()

        # -----------------------------------------------------------
        # 5. Clone request for route matching debug
        # -----------------------------------------------------------
        try:
            endpoint = request.scope.get("endpoint")
            print(f"[RID {rid}] Routed to endpoint: {endpoint}")
        except Exception:
            print(f"[RID {rid}] Could not determine endpoint.")

        # -----------------------------------------------------------
        # 6. Call downstream, but capture ANY exception
        # -----------------------------------------------------------
        try:
            response = await call_next(request)
            caught_exc = None
        except Exception as e:
            caught_exc = e
            response = Response(
                content=json.dumps({"detail": str(e)}),
                media_type="application/json",
                status_code=500,
            )

        duration_ms = round((time.time() - start) * 1000, 2)

        # -----------------------------------------------------------
        # 7. Capture FULL response body
        # -----------------------------------------------------------
        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk

        # Reset iterator so the response still sends to client
        response.body_iterator = iter([resp_body])

        resp_text = resp_body.decode("utf-8", errors="replace")

        try:
            resp_json = json.loads(resp_text)
        except Exception:
            resp_json = None

        # -----------------------------------------------------------
        # 8. Print response block
        # -----------------------------------------------------------
        print(f"[RID {rid}] <<< RESPONSE SENT <<<")
        print(f"Status: {response.status_code}")
        print(f"Time:   {duration_ms} ms")
        print("Response Body (raw):", resp_text)
        if resp_json is not None:
            print("Response Body (json):", json.dumps(resp_json, indent=2))

        # -----------------------------------------------------------
        # 9. Log exceptions separately for clarity
        # -----------------------------------------------------------
        if caught_exc:
            print(f"[RID {rid}] !!! EXCEPTION CAUGHT !!!")
            print("Exception:", repr(caught_exc))

        print("──────────────────────────────────────────────────────────\n")

        return response
