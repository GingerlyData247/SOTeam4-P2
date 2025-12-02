import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("request_logger")

class RequestResponseLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log incoming request
        body_bytes = await request.body()
        logger.info(
            "[REQUEST] method=%s path=%s query=%s body=%s",
            request.method,
            request.url.path,
            request.url.query,
            body_bytes.decode("utf-8", errors="ignore")
        )

        # Process request
        response = await call_next(request)

        # Get response body
        try:
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            logger.info(
                "[RESPONSE] status=%s path=%s body=%s",
                response.status_code,
                request.url.path,
                response_body.decode("utf-8", errors="ignore")
            )

            # Rebuild the iterator after consuming it
            response.body_iterator = iter([response_body])
        except Exception as e:
            logger.warning("Failed to log response body: %s", e)

        return response
