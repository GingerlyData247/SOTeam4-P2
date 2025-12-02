import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("request_logger")
logging.basicConfig(level=logging.INFO)


class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        # ---- Log request ----
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8") if body_bytes else ""
        except Exception:
            body_text = "<unreadable>"

        logger.info(
            f"[REQUEST] method={request.method} path={request.url.path} "
            f"query={request.url.query} body={body_text}"
        )

        # ---- Get response ----
        response = await call_next(request)

        # ---- CANNOT READ body_iterator (breaks streaming / Mangum)
        # So only log status + path
        logger.info(
            f"[RESPONSE] status={response.status_code} path={request.url.path}"
        )

        return response
