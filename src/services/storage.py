# src/services/storage.py
from __future__ import annotations
import os
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError


# ---------------------------------------------------------
# LOCAL FALLBACK STORAGE (used when S3_BUCKET is not set)
# ---------------------------------------------------------
class LocalStorage:
    """
    Local dummy storage for development.
    Stores files under ./local_storage/ and returns local:// URLs.
    No AWS charges.
    """
    BASE = "local_storage"

    def __init__(self):
        os.makedirs(self.BASE, exist_ok=True)

    def put_bytes(self, key: str, data: bytes) -> str:
        path = os.path.join(self.BASE, key.replace("/", "_"))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            f.write(data)

        return f"local://{path}"

    def presign(self, key: str) -> str:
        # For local mode, the presigned URL is just a readable path.
        return f"local://download/{key.replace('/', '_')}"


# ---------------------------------------------------------
# REAL S3 STORAGE (used when S3_BUCKET is set)
# ---------------------------------------------------------
class S3Storage:
    def __init__(self, bucket: str, region: str | None):
        self.bucket = bucket
        self.region = region
        self.client = boto3.client("s3", region_name=region)

    def put_bytes(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def presign(self, key: str) -> str:
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=3600,
        )


# ---------------------------------------------------------
# FACTORY FUNCTION
# ---------------------------------------------------------
_storage_instance = None


def get_storage():
    """
    Returns S3Storage when S3_BUCKET exists.
    Falls back to LocalStorage when running locally.
    """
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("AWS_REGION")

    if bucket:
        # Running in Lambda or S3-enabled environment
        _storage_instance = S3Storage(bucket, region)
        print(f"[storage] Using S3 backend (bucket={bucket})")
    else:
        # Local fallback — NO AWS usage, zero cost
        _storage_instance = LocalStorage()
        print("[storage] Using LOCAL storage backend")

    return _storage_instance
