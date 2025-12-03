import logging
import json
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("deep_logger")
logging.basicConfig(level=logging.INFO)

print("### LAMBDA-SAFE LOGGER LOADED ###", __file__)


class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = str(uuid.uuid4())[:8]
        start = time.time()

        # -----------------------------------------------------------
        # Capture request body
        # -----------------------------------------------------------
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8", errors="replace")
        except:
            body_bytes = b""
            body_text = "<unreadable>"

        try:
            req_json = json.loads(body_text)
        except:
            req_json = None

        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[RID {rid}] >>> REQUEST >>>")
        print(f"Method: {request.method}")
        print(f"Path:   {request.url.path}")
        print(f"Query:  {request.url.query}")
        print("Headers:")
        for k, v in request.headers.items():
            print(f"  {k}: {v}")
        print("Body (raw):", body_text)
        if req_json is not None:
            print("Body (json):", json.dumps(req_json, indent=2))

        # -----------------------------------------------------------
        # Execute route handler
        # -----------------------------------------------------------
        response = await call_next(request)

        # -----------------------------------------------------------
        # Capture response body (Lambda-safe)
        # -----------------------------------------------------------
        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk
        response.body_iterator = iter([resp_body])

        resp_text = resp_body.decode("utf-8", errors="replace")
        try:
            resp_json = json.loads(resp_text)
        except:
            resp_json = None

        # -----------------------------------------------------------
        # Print logs
        # -----------------------------------------------------------
        duration_ms = round((time.time() - start) * 1000, 2)
        print(f"[RID {rid}] <<< RESPONSE <<<")
        print("Status:", response.status_code)
        print("Time:", duration_ms, "ms")
        print("Response Body (raw):", resp_text)
        if resp_json is not None:
            print("Response Body (json):", json.dumps(resp_json, indent=2))
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        return response
