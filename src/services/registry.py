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

        # 1) Locate root model
        root = self.get(id_)
        if root is None:
            raise KeyError(f"Model {id_} not found")

        # 2) Build helper maps
        id_map: Dict[str, Dict[str, Any]] = {}
        name_map: Dict[str, Dict[str, Any]] = {}
        parents_of: Dict[str, List[str]] = {}

        for m in self._models:
            # SAFETY CHECK: Skip malformed entries
            if not isinstance(m, dict): continue
            
            # Safely extract ID and Name
            mid = str(m.get("id", ""))
            name = m.get("name")
            if not mid or not name: continue

            id_map[mid] = m
            name_map[name] = m

            # Safely extract parents
            meta = m.get("metadata")
            if not isinstance(meta, dict): 
                meta = {}
            
            parents = meta.get("parents")
            if not isinstance(parents, list):
                parents = []

            # Normalize parents to list of strings
            parents_of[mid] = [str(p) for p in parents if p]

        # 3) Build children map
        children_of: Dict[str, List[str]] = {}
        for child_id, parent_names in parents_of.items():
            for pname in parent_names:
                if pname in name_map:
                    parent_model = name_map[pname]
                    pid = str(parent_model["id"])
                    children_of.setdefault(pid, []).append(child_id)

        # 4) BFS
        visited = set()
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, str]] = []

        root_id = str(root["id"])
        queue: List[str] = [root_id]
        visited.add(root_id)

        while queue:
            cur_id = queue.pop(0)
            if cur_id not in id_map: continue
                
            cur_model = id_map[cur_id]
            nodes[cur_id] = {"id": cur_id, "name": cur_model["name"]}

            # Process Parents
            current_parents = parents_of.get(cur_id, [])
            for pname in current_parents:
                if pname in name_map:
                    p_model = name_map[pname]
                    pid = str(p_model["id"])
                    
                    nodes[pid] = {"id": pid, "name": p_model["name"]}
                    edges.append({"parent": pid, "child": cur_id})
                    
                    if pid not in visited:
                        visited.add(pid)
                        queue.append(pid)

            # Process Children
            current_children = children_of.get(cur_id, [])
            for child_id in current_children:
                if child_id in id_map:
                    c_model = id_map[child_id]
                    
                    nodes[child_id] = {"id": child_id, "name": c_model["name"]}
                    edges.append({"parent": cur_id, "child": child_id})
                    
                    if child_id not in visited:
                        visited.add(child_id)
                        queue.append(child_id)

        return {
            "root_id": root_id,
            "nodes": list(nodes.values()),
            "edges": edges,
        }
