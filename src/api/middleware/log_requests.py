import json
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger("request_logger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # ---------------- REQUEST ----------------
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8") if body_bytes else ""
        except Exception:
            body_text = "<unreadable>"

        logger.info(
            f"[REQUEST] method={request.method} path={request.url.path} "
            f"query={request.url.query} body={body_text}"
        )

        # ---------------- RESPONSE ----------------
        response = await call_next(request)

        body = b""

        # Try async iterator first
        try:
            async for chunk in response.body_iterator:
                body += chunk
        except TypeError:
            # If it's a normal iterator (Swagger docs, errors, JSON)
            try:
                for chunk in response.body_iterator:
                    body += chunk
            except Exception:
                body = b"<unreadable>"

        # Rebuild the response (important)
        new_response = Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        try:
            body_text = body.decode("utf-8")
        except Exception:
            body_text = "<binary>"

        logger.info(
            f"[RESPONSE] status={response.status_code} "
            f"path={request.url.path} body={body_text}"
        )

        return new_response
