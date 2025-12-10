import json
import boto3
from typing import List, Dict, Any, Optional

class RegistryService:
    def __init__(self, bucket_name: str, key: str = "registry/registry.json"):
        self.s3 = boto3.client("s3")
        self.bucket = bucket_name
        self.key = key

        # Local state (always loaded at start of each request)
        self._models: List[Dict[str, Any]] = []
        self._id_counter: int = 0

        self._load()

    # -----------------------------
    # Internal S3 helpers
    # -----------------------------
    def _load(self):
        """Load registry.json from S3."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            data = json.loads(obj["Body"].read())
            self._models = data.get("models", [])
            self._id_counter = data.get("id_counter", 0)
        except self.s3.exceptions.NoSuchKey:
            # Initialize on first run
            self._models = []
            self._id_counter = 0
            self._save()

    def _save(self):
        """Write registry.json back to S3."""
        data = {
            "models": self._models,
            "id_counter": self._id_counter,
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.key,
            Body=json.dumps(data).encode("utf-8"),
            ContentType="application/json",
        )

    # -----------------------------
    # API used by your project
    # -----------------------------
    def create(self, m) -> Dict[str, Any]:
        self._load()  # ensure fresh state
        self._id_counter += 1
        new_id = str(self._id_counter)

        meta = dict(m.metadata) if m.metadata else {}

        entry = {
            "id": new_id,
            "name": m.name,
            "version": m.version,
            "metadata": meta,
        }

        self._models.append(entry)
        self._save()
        return entry

    def get(self, id_: str) -> Optional[Dict[str, Any]]:
        self._load()

        if id_ is None:
            return None

        # Normalize input
        id_str = str(id_).strip()

        for m in self._models:
            mid = str(m.get("id", "")).strip()
            if mid == id_str:
                return m

        return None

    def delete(self, id_: str) -> bool:
        self._load()
        before = len(self._models)
        self._models = [m for m in self._models if m["id"] != id_]
        if len(self._models) < before:
            self._save()
            return True
        return False

    def count_models(self) -> int:
        self._load()
        return len(self._models)

    def reset(self) -> None:
        self._models = []
        self._id_counter = 0
        self._save()
