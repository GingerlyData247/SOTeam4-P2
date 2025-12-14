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
        except Exception:
            # Initialize on first run or error
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
        
        # Ensure 'parents' is preserved if present in metadata
        if "parents" not in meta and hasattr(m, "metadata") and "parents" in m.metadata:
             meta["parents"] = m.metadata["parents"]

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

    # ------------------------------------------------------------------ #
    # LINEAGE GRAPH (Added back to support the Lineage button)
    # ------------------------------------------------------------------ #
    def get_lineage_graph(self, id_: str) -> Dict[str, Any]:
        """
        Build a lineage graph rooted at the model with id `id_`.
        """
        self._load() # Ensure we have the latest data

        # 1) Locate root model
        root = self.get(id_)
        if root is None:
            raise KeyError(f"Model {id_} not found")

        # 2) Build helper maps for quick lookups
        all_models: List[Dict[str, Any]] = list(self._models)
        id_map: Dict[str, Dict[str, Any]] = {str(m["id"]): m for m in all_models}
        name_map: Dict[str, Dict[str, Any]] = {m["name"]: m for m in all_models}

        # parents_of[model_id] -> list of parent *names* (strings)
        parents_of: Dict[str, List[str]] = {}
        for m in all_models:
            mid = str(m["id"])
            meta = m.get("metadata") or {}
            parents = meta.get("parents")
            
            # FIX: Ensure parents is strictly a list before iterating
            if not isinstance(parents, list):
                parents = []

            # normalize: ensure it's always a list of strings
            parents_of[mid] = [
                p for p in parents if isinstance(p, str) and p
            ]

        # 3) Reverse map: children_of[parent_id] -> list of child_ids
        children_of: Dict[str, List[str]] = {}
        for child_id, parent_names in parents_of.items():
            for pname in parent_names:
                parent_model = name_map.get(pname)
                if not parent_model:
                    continue
                pid = str(parent_model["id"])
                children_of.setdefault(pid, []).append(child_id)

        # 4) BFS starting from root
        visited = set()
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, str]] = []

        root_id = str(root["id"])
        queue: List[str] = [root_id]
        visited.add(root_id)

        while queue:
            cur_id = queue.pop(0)
            if cur_id not in id_map:
                continue
                
            cur_model = id_map[cur_id]
            nodes[cur_id] = {"id": cur_id, "name": cur_model["name"]}

            # Parent edges
            for pname in parents_of.get(cur_id, []):
                p_model = name_map.get(pname)
                if not p_model:
                    continue
                pid = str(p_model["id"])
                
                if pid not in id_map: 
                    continue

                nodes[pid] = {"id": pid, "name": p_model["name"]}
                edges.append({"parent": pid, "child": cur_id})
                if pid not in visited:
                    visited.add(pid)
                    queue.append(pid)

            # Child edges
            for child_id in children_of.get(cur_id, []):
                if child_id not in id_map:
                    continue
                    
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
