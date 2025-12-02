import logging
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("request_logger")
logger.setLevel(logging.INFO)

# Attach a stream handler explicitly (Lambda requires this)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Request log
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8") if body_bytes else ""

        logger.info(f"[REQUEST] {request.method} {request.url.path}?{request.url.query} body={body_text}")

        response = await call_next(request)

        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        response.body_iterator = iter([response_body])

        try:
            body_str = response_body.decode("utf-8")
        except:
            body_str = "<binary>"

        logger.info(f"[RESPONSE] {response.status_code} {request.url.path} body={body_str}")

        return response
