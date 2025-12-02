import logging
import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("request_logger")
logging.basicConfig(level=logging.INFO)

print("### LOG MIDDLEWARE LOADED ###", __file__)

class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        # ---- Basic request info ----
        method = request.method
        path = request.url.path
        query = request.url.query
        client = request.client.host if request.client else "unknown"

        # ---- Read request body safely ----
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            body_text = "<unreadable>"

        # Try to JSON parse (optional)
        try:
            parsed_json = json.loads(body_text)
        except Exception:
            parsed_json = None

        # ---- Print EVERYTHING to CloudWatch ----
        print("─────────────────────────────────────────────")
        print(f"### REQUEST RECEIVED ###")
        print(f"Client: {client}")
        print(f"Method: {method}")
        print(f"Path:   {path}")
        print(f"Query:  {query}")
        print(f"Headers:")
        for k, v in request.headers.items():
            print(f"  {k}: {v}")

        print("Body (raw):", body_text)
        if parsed_json is not None:
            print("Body (JSON):", json.dumps(parsed_json, indent=2))

        print("─────────────────────────────────────────────")

        # ---- Call downstream route ----
        response = await call_next(request)

        # ---- Log response status ----
        print(f"### RESPONSE SENT ### status={response.status_code} path={path}")
        print("─────────────────────────────────────────────")

        return response
