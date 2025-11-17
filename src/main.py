# src/main.py
import os
from dotenv import load_dotenv
load_dotenv()  # load .env early

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from src.api.routers import models as models_router
from src.api.routers.tracks import router as tracks_router
from src.api.routes_s3 import router as s3_router  # mounts /api/s3/*

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

# Mount routers under /api
app.include_router(models_router.router, prefix="/api")
app.include_router(tracks_router.router, prefix="/api")
app.include_router(s3_router)  # routes_s3 defines prefix="/api/s3"

# Health and env endpoints
@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/env")
def get_env_values():
    return {
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "AWS_REGION": os.getenv("AWS_REGION"),
    }

# Single Lambda entrypoint
handler = Mangum(app)
