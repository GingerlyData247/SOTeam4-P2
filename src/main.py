# src/main.py
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Response
from starlette.middleware.cors import CORSMiddleware
from mangum import Mangum

from src.api.routers.models import router as models_router
from src.api.routes_s3 import router as s3_router
from src.api.middleware.log_requests import DeepASGILogger

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
FRONTEND_ORIGIN = "http://sot4-model-registry-dev.s3-website.us-east-2.amazonaws.com"
ALLOWED_ORIGINS = [FRONTEND_ORIGIN]

# -------------------------------------------------------------
# Create the FastAPI app FIRST
# -------------------------------------------------------------
app = FastAPI(title="SOTeam4P2 API")

print(">>> MIDDLEWARE ACTIVE <<<")

# -------------------------------------------------------------
# Add middleware SECOND
# -------------------------------------------------------------
app.add_middleware(DeepASGILogger)

# CORS middleware (backend is source of truth; API Gateway CORS is cleared)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Include Routers THIRD
# -------------------------------------------------------------
app.include_router(models_router, prefix="/api")
app.include_router(s3_router)

# -------------------------------------------------------------
# Global preflight handler (prevents OPTIONS -> 500 and guarantees headers)
# -------------------------------------------------------------
@app.options("/{path:path}")
async def preflight_handler(path: str):
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": FRONTEND_ORIGIN,
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "content-type,authorization",
        },
    )

# -------------------------------------------------------------
# Extra debugging endpoint
# -------------------------------------------------------------
@app.get("/env")
def get_env_values():
    return {
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "AWS_REGION": os.getenv("AWS_REGION"),
    }

# -------------------------------------------------------------
# Create Lambda handler LAST
# -------------------------------------------------------------
handler = Mangum(app)
