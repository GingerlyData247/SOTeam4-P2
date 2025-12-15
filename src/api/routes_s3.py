# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: S3 API ROUTES
# REQUIREMENTS SATISFIED: backend storage support for artifact data and uploads

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/api/routes_s3.py

Defines API routes for interacting with Amazon S3 storage.

This module provides lightweight FastAPI endpoints that expose basic
object storage functionality to the backend. These routes are used to
store and retrieve text data and uploaded files, supporting artifact
storage, debugging, and deployment workflows within the Trustworthy
Model Registry system.

Endpoints:
    - POST /api/s3/put-text   : Store UTF-8 text at a specified key
    - GET  /api/s3/get-text   : Retrieve stored text by key
    - POST /api/s3/upload    : Upload an arbitrary file to S3

The routes act as a thin abstraction over S3 utilities and are not
intended to be directly user-facing. All operations are designed to
fail safely with clear HTTP error responses.
"""
# src/api/routes_s3.py
from fastapi import APIRouter, UploadFile, HTTPException, Body
from src.aws.s3_utils import upload_to_s3, download_from_s3

router = APIRouter(prefix="/api/s3", tags=["S3"])

@router.post("/put-text")
async def put_text(key: str, body: str = Body(...)):
    path = upload_to_s3(key, body.encode("utf-8"))
    return {"ok": True, "path": path}

@router.get("/get-text")
def get_text(key: str):
    try:
        data = download_from_s3(key)
        return {"ok": True, "key": key, "body": data.decode("utf-8")}
    except Exception:
        raise HTTPException(status_code=404, detail=f"not found: {key}")

@router.post("/upload")
async def upload_file(file: UploadFile):
    data = await file.read()
    path = upload_to_s3(f"uploads/{file.filename}", data)
    return {"ok": True, "path": path, "size": len(data)}
