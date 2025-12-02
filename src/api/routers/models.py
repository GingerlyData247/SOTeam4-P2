# src/api/routers/models.py

from __future__ import annotations

import io
import time
import zipfile
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Body, HTTPException, Path, Query, Response
from pydantic import BaseModel

from ...schemas.models import ModelCreate
from ...services.registry import RegistryService
from ...services.scoring import ScoringService
from ...services.storage import get_storage
from src.metrics.treescore import extract_parents_from_resource
from src.run import compute_metrics_for_model
from src.utils.hf_normalize import normalize_hf_id

_START_TIME = time.time()

router = APIRouter()

_registry = RegistryService()
_scoring = ScoringService()
_storage = get_storage()

# ---------------------------------------------------------------------------
# Simple schema helpers
# ---------------------------------------------------------------------------

ArtifactTypeLiteral = Literal["model", "dataset", "code"]

class ArtifactData(BaseModel):
    url: str
    download_url: Optional[str] = None

class ArtifactMetadata(BaseModel):
    name: str
    id: str
    type: ArtifactTypeLiteral

class Artifact(BaseModel):
    metadata: ArtifactMetadata
    data: ArtifactData

class ArtifactQuery(BaseModel):
    name: str
    types: Optional[List[ArtifactTypeLiteral]] = None

class ArtifactRegex(BaseModel):
    regex: str

class SimpleLicenseCheckRequest(BaseModel):
    github_url: str

class LicenseCheckInternalRequest(BaseModel):
    github_url: str

LICENSE_COMPATIBILITY: Dict[str, set] = {
    "apache-2.0": {"mit", "bsd-3-clause", "bsd-2-clause", "apache-2.0"},
    "mit": {"mit", "bsd-3-clause", "bsd-2-clause"},
    "bsd-3-clause": {"bsd-3-clause", "mit"},
    "bsd-2-clause": {"bsd-2-clause", "mit"},
    "gpl-3.0": set(),
    "cc-by-4.0": set(),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_id_from_url_or_id(s: str) -> str:
    return normalize_hf_id(s.strip())

def _build_hf_resource(hf_id: str) -> Dict[str, Any]:
    return _scoring._build_resource(hf_id)

def _extract_repo_info(url: str) -> (str, str):
    parsed = urlparse(url)
    if parsed.netloc not in ("github.com", "www.github.com"):
        raise ValueError("Invalid GitHub URL; expected https://github.com/<owner>/<repo>")
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL; expected https://github.com/<owner>/<repo>")
    return parts[0], parts[1]

def _fetch_github_license(owner: str, repo: str) -> str:
    api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
    resp = requests.get(api_url, headers={"Accept": "application/vnd.github+json"})
    if resp.status_code == 200:
        data = resp.json()
        return (data.get("license", {}) or {}).get("spdx_id", "").lower()
    raise ValueError("Unable to determine GitHub project license.")

def _fetch_hf_license(hf_id: str) -> str:
    api_url = f"https://huggingface.co/api/models/{hf_id}"
    resp = requests.get(api_url, timeout=10)
    if resp.status_code != 200:
        raise ValueError("Unable to fetch Hugging Face model metadata.")
    data = resp.json()
    return (data.get("license") or "").lower()

def _bytes_to_mb(n: int) -> float:
    return round(float(n) / 1_000_000.0, 3)

# ---------------------------------------------------------------------------
# Ingest logic
# ---------------------------------------------------------------------------

def _ingest_hf_core(source_url: str) -> Dict[str, Any]:
    hf_id = _hf_id_from_url_or_id(source_url)
    hf_url = f"https://huggingface.co/{hf_id}"

    try:
        hf_license = _fetch_hf_license(hf_id)
    except Exception:
        hf_license = ""

    base_resource = {
        "name": hf_id,
        "url": hf_url,
        "github_url": None,
        "local_path": None,
        "skip_repo_metrics": False,
        "category": "MODEL",
    }

    metrics = compute_metrics_for_model(base_resource)
    reviewedness = float(metrics.get("reviewedness", 0.0) or 0.0)
    if reviewedness < 0.5:
        raise HTTPException(
            status_code=424,
            detail=f"Ingest rejected: reviewedness={reviewedness:.2f} < 0.50",
        )

    try:
        enriched = _build_hf_resource(hf_id)
        parents = extract_parents_from_resource(enriched)
    except Exception:
        enriched = {}
        parents = []

    metrics["parents"] = parents
    metrics["license"] = hf_license

    mc = ModelCreate(
        name=hf_id,
        version="1.0.0",
        card=metrics.get("card_text", ""),
        tags=list(metrics.get("tags", [])),
        metadata=metrics,
        source_uri=hf_url,
    )
    created = _registry.create(mc)

    created.setdefault("metadata", {})
    created["metadata"]["artifact_type"] = "model"

    model_id = created["id"]

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("source_url.txt", hf_url)

    try:
        key = f"artifacts/model/{model_id}.zip"
        _storage.put_bytes(key, mem_zip.getvalue())
        presigned = _storage.presign(key)
    except Exception:
        presigned = f"local://download/artifacts/model/{model_id}.zip"

    created["metadata"]["download_url"] = presigned
    return created

# ---------------------------------------------------------------------------
# /health, /reset, /tracks
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_s": int(time.time() - _START_TIME),
        "models": _registry.count_models(),
    }

@router.delete("/reset", status_code=200)
def reset_system():
    _registry.reset()
    try:
        _scoring.reset()
    except Exception:
        pass
    return {"status": "registry reset"}

@router.get("/tracks")
def get_tracks():
    return {"plannedTracks": ["Performance track"]}

# =======================================================================
# 🔥 STATIC ROUTES FIRST — critical fix (regex, byName, model/*)
# =======================================================================

# -----------------------
# POST /artifact/byRegEx
# -----------------------
@router.post("/artifact/byRegEx", response_model=List[ArtifactMetadata])
def artifact_by_regex(body: ArtifactRegex):
    import re
    try:
        pattern = re.compile(body.regex)
    except re.error:
        raise HTTPException(status_code=400, detail="Invalid regular expression.")
    all_items = list(_registry._models)

    def infer_type(entry: Dict[str, Any]) -> ArtifactTypeLiteral:
        meta = entry.get("metadata") or {}
        t = str(meta.get("artifact_type") or "").lower()
        return t if t in ("model", "dataset", "code") else "model"

    matches = []
    for m in all_items:
        name = m.get("name", "") or ""
        card = str((m.get("metadata") or {}).get("card", "") or "")
        if pattern.search(name) or pattern.search(card):
            matches.append(m)

    def sort_key(entry):
        id_val = entry.get("id", "")
        try:
            return int(id_val)
        except:
            return str(id_val)

    matches_sorted = sorted(matches, key=sort_key)

    return [
        ArtifactMetadata(name=m["name"], id=m["id"], type=infer_type(m))
        for m in matches_sorted
    ]

# -----------------------
# GET /artifact/byName/{name}
# -----------------------
@router.get("/artifact/byName/{name}", response_model=List[ArtifactMetadata])
def artifact_by_name(name: str):
    all_items = list(_registry._models)
    matches = [m for m in all_items if m.get("name") == name]
    if not matches:
        raise HTTPException(status_code=404, detail="No such artifact.")

    def infer_type(entry: Dict[str, Any]) -> ArtifactTypeLiteral:
        meta = entry.get("metadata") or {}
        t = str(meta.get("artifact_type") or "").lower()
        return t if t in ("model", "dataset", "code") else "model"

    def sort_key(entry):
        try:
            return int(entry["id"])
        except:
            return entry["id"]

    sorted_matches = sorted(matches, key=sort_key)

    return [
        ArtifactMetadata(name=m["name"], id=m["id"], type=infer_type(m))
        for m in sorted_matches
    ]

# -----------------------
# model/* static routes
# -----------------------


@router.get("/artifact/model/{id}/rate")
def model_artifact_rate(id: str):
    item = _registry.get(id)
    if not item:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    meta = item.get("metadata") or {}

    def g(name: str, alt: Optional[str] = None, default: float = 0.0) -> float:
        keys = (name, alt) if alt else (name,)
        for k in keys:
            v = meta.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return default

    size_metric = meta.get("size_score") or meta.get("size") or {}
    size_latency = g("size_score_latency", alt="size_latency")
    tree_score = g("tree_score", alt="treescore")
    tree_latency = g("tree_score_latency", alt="treescore_latency")

    return {
        "name": meta.get("name") or item["name"],
        "category": meta.get("category", ""),
        "net_score": g("net_score"),
        "net_score_latency": g("net_score_latency"),
        "ramp_up_time": g("ramp_up_time"),
        "ramp_up_time_latency": g("ramp_up_time_latency"),
        "bus_factor": g("bus_factor"),
        "bus_factor_latency": g("bus_factor_latency"),
        "performance_claims": g("performance_claims"),
        "performance_claims_latency": g("performance_claims_latency"),
        "license": g("license"),
        "license_latency": g("license_latency"),
        "dataset_and_code_score": g("dataset_and_code_score"),
        "dataset_and_code_score_latency": g("dataset_and_code_score_latency"),
        "dataset_quality": g("dataset_quality"),
        "dataset_quality_latency": g("dataset_quality_latency"),
        "code_quality": g("code_quality"),
        "code_quality_latency": g("code_quality_latency"),
        "reproducibility": g("reproducibility"),
        "reproducibility_latency": g("reproducibility_latency"),
        "reviewedness": g("reviewedness"),
        "reviewedness_latency": g("reviewedness_latency"),
        "tree_score": tree_score,
        "tree_score_latency": tree_latency,
        "size_score": {
            "raspberry_pi": float(size_metric.get("raspberry_pi", 0.0)),
            "jetson_nano": float(size_metric.get("jetson_nano", 0.0)),
            "desktop_pc": float(size_metric.get("desktop_pc", 0.0)),
            "aws_server": float(size_metric.get("aws_server", 0.0)),
        },
        "size_score_latency": size_latency,
    }

@router.get("/artifact/model/{id}/lineage")
def artifact_lineage(id: str):
    try:
        g = _registry.get_lineage_graph(id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    nodes_out = [
        {
            "artifact_id": n["id"],
            "name": n["name"],
            "source": "config_json",
            "metadata": {},
        }
        for n in g.get("nodes", [])
    ]

    edges_out = [
        {
            "from_node_artifact_id": e["parent"],
            "to_node_artifact_id": e["child"],
            "relationship": "base_model",
        }
        for e in g.get("edges", [])
    ]

    return {"nodes": nodes_out, "edges": edges_out}

@router.post("/artifact/model/{id}/license-check")
def artifact_license_check(id: str, body: SimpleLicenseCheckRequest):
    item = _registry.get(id)
    if not item:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    hf_id = item["name"]
    meta = item.get("metadata") or {}
    model_license = str(meta.get("license") or "").lower()

    if not model_license:
        try:
            model_license = _fetch_hf_license(hf_id)
        except Exception:
            raise HTTPException(status_code=502, detail="Unable to fetch model license.")

    try:
        owner, repo = _extract_repo_info(body.github_url)
        github_license = _fetch_github_license(owner, repo)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=502, detail="Unable to fetch GitHub license.")

    return github_license in LICENSE_COMPATIBILITY.get(model_license, set())

# =======================================================================
# 🚨 PARAMETERIZED ROUTES COME LAST (critical ordering)
# =======================================================================

@router.post("/artifact/{artifact_type}", response_model=Artifact, status_code=201)
def artifact_create(
    artifact_type: ArtifactTypeLiteral = Path(..., description="Only 'model', 'dataset', and 'code' supported."),
    body: ArtifactData = Body(...)
):
    if artifact_type == "model":
        created = _ingest_hf_core(body.url)
        return Artifact(
            metadata=ArtifactMetadata(
                name=created["name"],
                id=created["id"],
                type="model",
            ),
            data=ArtifactData(
                url=body.url,
                download_url=created["metadata"].get("download_url"),
            ),
        )

    if artifact_type in ("dataset", "code"):
        parsed = urlparse(body.url)
        path = parsed.path.rstrip("/")
        name = path.split("/")[-1] or artifact_type

        mc = ModelCreate(
            name=name,
            version="1.0.0",
            card="",
            tags=[],
            metadata={},
            source_uri=body.url,
        )

        created = _registry.create(mc)
        created.setdefault("metadata", {})
        created["metadata"]["artifact_type"] = artifact_type

        return Artifact(
            metadata=ArtifactMetadata(name=name, id=created["id"], type=artifact_type),
            data=ArtifactData(url=body.url, download_url=None)
        )

    raise HTTPException(status_code=400, detail="Unsupported artifact_type.")

@router.get("/artifact/{artifact_type}/{id}", response_model=Artifact)
def artifact_get(artifact_type: ArtifactTypeLiteral, id: str):
    item = _registry.get(id)
    if not item:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    meta = item.get("metadata") or {}
    stored_type = str(meta.get("artifact_type") or "").lower()
    if stored_type not in ("model", "dataset", "code"):
        stored_type = artifact_type

    source_uri = meta.get("source_uri") or item["name"]

    if stored_type == "model":
        url = source_uri or f"https://huggingface.co/{item['name']}"
        download_url = meta.get("download_url")
    else:
        url = source_uri
        download_url = None

    return Artifact(
        metadata=ArtifactMetadata(name=item["name"], id=item["id"], type=stored_type),
        data=ArtifactData(url=url, download_url=download_url)
    )

@router.put("/artifact/{artifact_type}/{id}", response_model=Artifact)
def artifact_update(artifact_type: str, id: str, body: Artifact):
    if body.metadata.id != id:
        raise HTTPException(status_code=400, detail="Name/id mismatch in artifact update.")

    item = _registry.get(id)
    if not item:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    meta = item.setdefault("metadata", {})
    meta["source_uri"] = body.data.url
    download_url = meta.get("download_url")

    stored_type = str(meta.get("artifact_type") or "model").lower()
    if stored_type not in ("model", "dataset", "code"):
        stored_type = "model"

    return Artifact(
        metadata=ArtifactMetadata(name=item["name"], id=item["id"], type=stored_type),
        data=ArtifactData(url=meta["source_uri"], download_url=download_url)
    )

@router.delete("/artifact/{artifact_type}/{id}")
def artifact_delete(artifact_type: str, id: str):
    ok = _registry.delete(id)
    if not ok:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    return {"status": "deleted", "id": id}

# ---------------------------------------------------------------------------
# POST /artifacts (baseline)
# ---------------------------------------------------------------------------

@router.post("/artifacts", response_model=List[ArtifactMetadata])
def artifacts_list(
    queries: List[ArtifactQuery],
    response: Response,
    offset: Optional[str] = Query(None)
):
    def infer_type(entry: Dict[str, Any]) -> ArtifactTypeLiteral:
        meta = entry.get("metadata") or {}
        t = str(meta.get("artifact_type") or "").lower()
        return t if t in ("model", "dataset", "code") else "model"

    q = queries[0] if queries else ArtifactQuery(name="*", types=None)
    all_items = list(_registry._models)

    if q.name == "*" or not q.name:
        name_filtered = all_items
    else:
        name_filtered = [m for m in all_items if m.get("name") == q.name]

    if q.types:
        allowed = set(q.types)
        type_filtered = [m for m in name_filtered if infer_type(m) in allowed]
    else:
        type_filtered = name_filtered

    start = 0
    if offset:
        try:
            start = int(offset)
        except:
            start = 0

    page_size = 1000
    slice_ = type_filtered[start : start + page_size]
    next_offset = start + page_size if (start + page_size) < len(type_filtered) else None

    if next_offset is not None:
        response.headers["offset"] = str(next_offset)

    return [
        ArtifactMetadata(name=m["name"], id=m["id"], type=infer_type(m))
        for m in slice_
    ]

@router.get("/artifact/model/{id}")
def artifact_model_get_stub(id: str):
    return {"detail": "Use /artifact/model/{id}/rate for model ratings."}
