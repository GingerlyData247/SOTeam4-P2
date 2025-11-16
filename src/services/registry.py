from __future__ import annotations
from typing import List, Dict, Any
import uuid

class RegistryService:
    def __init__(self):
        # Internal in-memory list
        self._models: List[Dict[str, Any]] = []

    def create(self, m):
        entry = {
            "id": str(uuid.uuid4()),
            "name": m.name,
            "version": m.version,
            "metadata": m.metadata or {
                "card": getattr(m, "card", ""),
                "tags": getattr(m, "tags", []),
                "source_uri": getattr(m, "source_uri", None),
            },
        }
        self._models.append(entry)
        return entry

    def list(self, q=None, limit=20, cursor=None):
        return {
            "items": self._models[:limit],
            "next_cursor": None,
        }

    def get(self, id_: str):
        return next((m for m in self._models if m["id"] == id_), None)

    def update(self, id_: str, m):
        for model in self._models:
            if model["id"] == id_:
                if m.description is not None:
                    model["metadata"]["description"] = m.description
                if m.tags is not None:
                    model["metadata"]["tags"] = m.tags
                return model
        return None

    def delete(self, id_: str):
        before = len(self._models)
        self._models = [m for m in self._models if m["id"] != id_]
        return len(self._models) < before

    def count_models(self):
        return len(self._models)

    def reset(self):
        self._models = []

