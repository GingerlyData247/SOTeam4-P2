from __future__ import annotations
from typing import Protocol

class Storage(Protocol):
    def put_object(self, key: str, bytes_or_path) -> str: ...
    def get_presigned_url(self, key: str, mode: str) -> str: ...

class LocalStorage:
    """Stub that echoes file keys; swap to S3 later without changing services."""
    def put_object(self, key: str, bytes_or_path) -> str:
        return f"local://{key}"
    def get_presigned_url(self, key: str, mode: str) -> str:
        return f"local-presigned://{mode}/{key}"

# Later: class S3Storage(Storage): ... (boto3)
