# src/main.py
import os
from dotenv import load_dotenv

load_dotenv()  # load .env early

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# IMPORTANT: import ONLY the router, not the module
from src.api.routers.models import router as models_router
from src.api.routes_s3 import router as s3_router


app = FastAPI(title="SOTeam4P2 API")

# --- CORS setup ---
origins = [
    "http://sot4-model-registry-dev.s3-website.us-east-2.amazonaws.com",
    "https://sot4-model-registry-dev.s3-website.us-east-2.amazonaws.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # wide-open for debugging; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Mount Routers
# -------------------------------------------------------------

# Mount ALL model-related endpoints under /api
# This gives:
#   /api/models/reset
#   /api/artifacts/reset
#   /api/models
#   /api/rate/{model}
#   /api/ingest
#   /api/health
#   /api/tracks
app.include_router(models_router, prefix="/api")

# Mount S3 routes (these already include their own /api/s3 prefix)
app.include_router(s3_router)

# -------------------------------------------------------------
# Environment debugging endpoint
# -------------------------------------------------------------
@app.get("/api/env")
def get_env_values():
    return {
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "AWS_REGION": os.getenv("AWS_REGION"),
    }

# Lambda handler
handler = Mangum(app)


# Single Lambda entrypoint
handler = Mangum(app)
