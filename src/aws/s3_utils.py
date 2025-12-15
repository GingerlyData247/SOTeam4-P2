# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: S3 STORAGE UTILITIES
# REQUIREMENTS SATISFIED: artifact storage and retrieval support

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/aws/s3_utils.py

Provides helper utilities for interacting with Amazon S3 storage.

This module encapsulates common S3 operations used throughout the backend,
including uploading and downloading raw byte data. It centralizes S3
client creation and bucket configuration, allowing the rest of the
application to remain agnostic to AWS environment details.

Key features:
    - Automatically resolves the S3 bucket name from environment variables
    - Creates S3 clients compatible with both AWS Lambda and local execution
    - Supports uploading and downloading arbitrary byte data
    - Exposes compatibility aliases for legacy storage interfaces

These utilities are used by API routes and backend services to persist
artifact data and generated files in a reliable, cloud-backed storage
layer.
"""
# src/aws/s3_utils.py
import os
import boto3

def _bucket() -> str:
    b = os.getenv("S3_BUCKET")
    if not b:
        raise RuntimeError("S3_BUCKET not set")
    return b

def _client():
    """
    Create an S3 client.
    - In Lambda: boto3 auto-detects region from the runtime env (AWS_REGION).
    - Locally: use AWS config or the AWS_REGION env var if present.
    """
    region = os.getenv("AWS_REGION")  # present in Lambda; optional locally
    return boto3.client("s3", region_name=region) if region else boto3.client("s3")

def upload_to_s3(key: str, data: bytes) -> str:
    """Upload bytes to S3 and return s3:// URI."""
    client = _client()
    bucket = _bucket()
    client.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"

def download_from_s3(key: str) -> bytes:
    """Download bytes from S3."""
    client = _client()
    bucket = _bucket()
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

# Compatibility aliases
put_bytes = upload_to_s3
get_bytes = download_from_s3
