# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: REGISTRY SERVICE
# REQUIREMENTS SATISFIED: artifact persistence, lookup, and lineage graph support

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/services/registry.py

Defines the registry service responsible for persisting and managing
artifact records.

This module implements a lightweight registry backed by Amazon S3. It
stores artifact metadata, assigns unique identifiers, and provides CRUD
operations used throughout the backend. The service also supports
construction of simple lineage graphs for model artifacts based on
stored metadata.

Key responsibilities:
    - Persist artifact metadata to S3 using a JSON-based registry file
    - Assign and manage unique artifact identifiers
    - Provide create, retrieve, delete, count, and reset operations
    - Safely recover from S3 or JSON failures without crashing the system
    - Construct lineage graphs for model artifacts with external fallbacks

The registry service acts as the authoritative source of truth for
artifacts in the Trustworthy Model Registry and is designed to be
fault-tolerant and compatible with Phase 2 API requirements.
"""
import json
import boto3
import logging
from typing import List, Dict, Any, Optional

# Set up simple logging to help debug in CloudWatch
logger = logging.getLogger("registry_service")
logger.setLevel(logging.INFO)

class RegistryService:
    def __init__(self, bucket_name: str, key: str = "registry/registry.json"):
        self.s3 = boto3.client("s3")
        self.bucket = bucket_name
        self.key = key

        # Local state
        self._models: List[Dict[str, Any]] = []
        self._id_counter: int = 0

        self._load()

    # -----------------------------
    # Internal S3 helpers
    # -----------------------------
    def _load(self):
        """Load registry.json from S3 safely."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            content = obj["Body"].read()
            data = json.loads(content)
            self._models = data.get("models", [])
            self._id_counter = data.get("id_counter", 0)
        except Exception as e:
            # If S3 fails or JSON is invalid, start empty to prevent 500s
            logger.error(f"Failed to load registry: {e}")
            self._models = []
            self._id_counter = 0

    def _save(self):
        """Write registry.json back to S3."""
        try:
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
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    # -----------------------------
    # CRUD API
    # -----------------------------
    def create(self, m) -> Dict[str, Any]:
        self._load()
        self._id_counter += 1
        new_id = str(self._id_counter)

        # Handle metadata safely
        meta = {}
        if hasattr(m, "metadata") and isinstance(m.metadata, dict):
            meta = dict(m.metadata)
        
        # Ensure 'parents' is preserved if present
        if "parents" not in meta and hasattr(m, "metadata") and isinstance(m.metadata, dict):
             if "parents" in m.metadata:
                 meta["parents"] = m.metadata["parents"]

        entry = {
            "id": new_id,
            "name": getattr(m, "name", "Unnamed Model"),
            "version": getattr(m, "version", "1.0.0"),
            "metadata": meta,
        }

        self._models.append(entry)
        self._save()
        return entry

    def get(self, id_: str) -> Optional[Dict[str, Any]]:
        self._load()
        if id_ is None:
            return None
        
        id_str = str(id_).strip()
        for m in self._models:
            # Safely get ID from model entry
            mid = str(m.get("id", "")).strip()
            if mid == id_str:
                return m
        return None

    def delete(self, id_: str) -> bool:
        self._load()
        before = len(self._models)
        self._models = [m for m in self._models if str(m.get("id", "")) != id_]
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

    # ------------------------------------------------------------------ #
    # LINEAGE GRAPH (Hardened against bad data)
    # ------------------------------------------------------------------ #
    def get_lineage_graph(self, id_: str) -> Dict[str, Any]:
        self._load()
    
        root = self.get(id_)
        if root is None:
            raise KeyError(f"Model {id_} not found")
    
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, str]] = []
    
        root_id = str(root["id"])
        root_name = root["name"]
    
        # --- Always include root node FIRST (frontend requirement)
        nodes[root_id] = {
            "artifact_id": root_id,
            "name": root_name,
            "source": "config_json",
            "metadata": {}
        }
    
        # --- 1. Extract HF parent if present
        meta = root.get("metadata") or {}
        hf_parent = meta.get("base_model_name_or_path")
    
        if hf_parent:
            hf_parent = hf_parent.replace("https://huggingface.co/", "").strip()
    
            # --- 2. Check registry
            parent_model = None
            for m in self._models:
                if m.get("name") == hf_parent:
                    parent_model = m
                    break
    
            if parent_model:
                pid = str(parent_model["id"])
    
                nodes[pid] = {
                    "artifact_id": pid,
                    "name": parent_model["name"],
                    "source": "config_json",
                    "metadata": {}
                }
    
                edges.append({
                    "from_node_artifact_id": pid,
                    "to_node_artifact_id": root_id,
                    "relationship": "base_model"
                })
    
            else:
                # --- LEVEL 2: placeholder parent
                external_id = f"external:{hf_parent}"
    
                nodes[external_id] = {
                    "artifact_id": external_id,
                    "name": hf_parent,
                    "source": "config_json",
                    "metadata": {"external": True}
                }
    
                edges.append({
                    "from_node_artifact_id": external_id,
                    "to_node_artifact_id": root_id,
                    "relationship": "base_model"
                })
    
        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }

