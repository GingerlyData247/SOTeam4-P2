"""import logging
from starlette.types import ASGIApp, Scope, Receive, Send

logger = logging.getLogger("request_logger")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class RequestResponseLogger:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]
        query = scope.get("query_string", b"").decode()

        logger.info(f"[REQUEST] {method} {path} ?{query}")

        # Wrap send() to capture response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                logger.info(f"[RESPONSE] {status} {path}")
            await send(message)

        await self.app(scope, receive, send_wrapper)
"""
