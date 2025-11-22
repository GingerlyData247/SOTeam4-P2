import os
import boto3
from botocore.client import Config
from typing import Optional

# Check if using AWS or local mode
LOCAL_MODE = os.getenv("LOCAL_STORAGE", "0") == "1"

# Bucket name (required in AWS mode)
BUCKET = os.getenv("S3_BUCKET")

if not LOCAL_MODE and not BUCKET:
    raise RuntimeError("S3_BUCKET not set")

# S3 client
_s3 = None
def _client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            config=Config(signature_version="s3v4"),
        )
    return _s3


# -------- LOCAL STORAGE FALLBACK --------
LOCAL_DIR = "/tmp/local-artifacts"

def _local_write(key: str, data: bytes):
    path = os.path.join(LOCAL_DIR, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def _local_presign(key: str) -> str:
    return f"local://download/{key}"


# -------- PUBLIC API --------
class Storage:
    def put_bytes(self, key: str, data: bytes):
        """
        Store arbitrary bytes. Works for .zip, .bin, .txt — anything.
        """
        if LOCAL_MODE:
            return _local_write(key, data)

        return _client().put_object(Bucket=BUCKET, Key=key, Body=data)

    def presign(self, key: str, expires: int = 3600) -> str:
        """
        Generate a presigned S3 URL or local placeholder.
        """
        if LOCAL_MODE:
            return _local_presign(key)

        return _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=expires,
        )


_storage_instance = Storage()

def get_storage():
    return _storage_instance
