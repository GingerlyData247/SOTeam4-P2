# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: FASTAPI APPLICATION ENTRY POINT
# REQUIREMENTS SATISFIED:
#   - API initialization and routing
#   - Middleware configuration (logging + CORS)
#   - AWS Lambda compatibility via Mangum
#   - Baseline API availability and preflight handling

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/main.py

Primary application entry point for the Team 4 Phase 2 Trustworthy Model
Registry backend. This module is responsible for assembling the FastAPI
application, registering middleware, mounting API routers, and exposing
the AWS Lambda handler.

Execution Order (Intentional):
    1. Environment variables are loaded from .env
    2. FastAPI app is created
    3. Custom ASGI request/response logging middleware is attached
    4. CORS middleware is configured with a strict frontend allowlist
    5. API routers are mounted under the /api prefix
    6. A global OPTIONS handler is installed for CORS preflight safety
    7. The Mangum handler is created for AWS Lambda deployment

Key Design Decisions:
    - Middleware is added before routers to guarantee full request coverage.
    - A global OPTIONS handler prevents API Gateway preflight failures.
    - CORS configuration is handled at the application level to ensure
      consistent behavior across local, EC2, and Lambda deployments.
    - Mangum is used to adapt FastAPI to AWS Lambda without modifying
      application logic.

Deployment Context:
    - Designed to run both locally (uvicorn) and in AWS Lambda.
    - Uses environment variables for AWS configuration (S3 bucket, region).
    - Frontend is hosted as an S3 static website and explicitly allowlisted.

This file intentionally contains minimal business logic and serves only as
the orchestration layer for the backend system.
"""
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
