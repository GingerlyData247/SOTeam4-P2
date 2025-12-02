import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import iterate_in_threadpool

logger = logging.getLogger("request_logger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # --- Log Request ---
        try:
            body = await request.body()
            body_text = body.decode("utf-8") if body else ""
        except Exception:
            body_text = "<unreadable>"

        logger.info(f"[REQUEST] {request.method} {request.url.path}?{request.url.query} body={body_text}")

        # --- Process Response ---
        response = await call_next(request)

        # Read entire response body (async-friendly)
        raw_body = b""
        async for chunk in response.body_iterator:
            raw_body += chunk

        # Reset the async iterator properly (THIS FIXES THE CRASH)
        response.body_iterator = iterate_in_threadpool([raw_body])

        # Log Response
        try:
            decoded = raw_body.decode("utf-8")
        except Exception:
            decoded = "<binary>"

        logger.info(f"[RESPONSE] {response.status_code} {request.url.path} body={decoded}")

        return response
