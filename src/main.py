# src/main.py
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from mangum import Mangum

from src.api.routers.models import router as models_router
from src.api.routes_s3 import router as s3_router
from src.api.middleware.log_requests import DeepASGILogger

# -------------------------------------------------------------
# Create the FastAPI app FIRST
# -------------------------------------------------------------
app = FastAPI(title="SOTeam4P2 API")

print(">>> MIDDLEWARE ACTIVE <<<")

# -------------------------------------------------------------
# Add middleware SECOND
# -------------------------------------------------------------
app.add_middleware(DeepASGILogger)
# -------------------------------------------------------------
# Add CORS middleware
# -------------------------------------------------------------
ALLOWED_ORIGINS = ["http://sot4-model-registry-dev.s3-website.us-east-2.amazonaws.com"]
# -------------------------------------------------------------
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

@app.options("/{path:path}")
async def preflight(path: str, request: Request):
    return Response(status_code=204)

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
