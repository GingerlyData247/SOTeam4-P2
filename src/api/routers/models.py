# src/api/routers/models.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Header, Response
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from urllib.parse import urlparse
import requests
import time
from datetime import datetime, timezone

from ...schemas.models import ModelCreate, ModelUpdate, ModelOut, Page
from ...services.registry import RegistryService
from ...services.ingest import IngestService
from ...services.scoring import ScoringService

_START_TIME = time.time()

router = APIRouter()

_registry = RegistryService()
_ingest = IngestService(registry=_registry)
_scoring = ScoringService()


# ---------------------------------------------------------------------------
# Existing CRUD-style model endpoints (kept for your frontend)
# ---------------------------------------------------------------------------


@router.post("/models", response_model=ModelOut, status_code=201)
def create_model(body: ModelCreate) -> ModelOut:
    return _registry.create(body)


@router.get("/models", response_model=Page[ModelOut])
def list_models(
    q: Optional[str] = Query(default=None),
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = None,
) -> Page[ModelOut]:
    """
    Legacy /models listing endpoint used by your frontend.
    Not part of Phase 2 spec.
    """
    data = _registry.list(q=q, limit=limit, cursor=cursor)
    # Normalize to Page[...] shape
    return Page[ModelOut](items=data["items"], next_cursor=data.get("next"))


@router.get("/models/{model_id}", response_model=ModelOut)
def get_model(model_id: str) -> ModelOut:
    item = _registry.get(model_id)
    if not item:
        raise HTTPException(404, "Model not found")
    return item


@router.put("/models/{model_id}", response_model=ModelOut)
def update_model(model_id: str, body: ModelUpdate) -> ModelOut:
    updated = _registry.update(model_id, body)
    if not updated:
        raise HTTPException(404, "Model not found")
    return updated


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str):
    ok = _registry.delete(model_id)
    if not ok:
        raise HTTPException(404, "Model not found")


# ---------------------------------------------------------------------------
# Legacy /rate/{model_ref} endpoint (frontend "Get Score" button)
# ---------------------------------------------------------------------------


@router.get("/rate/{model_ref:path}")
def rate_model(model_ref: str):
    """
    Legacy Phase 1-style rating endpoint used by your frontend.

    Returns:
      {
        "net": float,
        "subs": { metric_name: score_or_object },
        "latency_ms": int
      }
    """
    from dotenv import load_dotenv
    from src.utils.hf_normalize import normalize_hf_id
    from src.run import compute_metrics_for_model
    import time as _time

    load_dotenv()

    start = _time.perf_counter()

    hf_id = normalize_hf_id(model_ref)
    hf_url = f"https://huggingface.co/{hf_id}"

    resource = {
        "name": hf_id,
        "url": hf_url,
        "github_url": None,
        "local_path": None,
        "skip_repo_metrics": False,
        "category": "MODEL",
    }

    result = compute_metrics_for_model(resource)

    latency_ms = int((_time.perf_counter() - start) * 1000)
    subs = {k: v for k, v in result.items() if isinstance(v, (float, dict))}

    return {
        "net": result.get("net_score", 0.0),
        "subs": subs,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Shared ingest helper (used by /ingest and spec /artifact/model)
# ---------------------------------------------------------------------------


def _ingest_model_from_ref(model_ref: str) -> Dict[str, Any]:
    """
    Core ingest logic: given a HF model ref or URL, compute metrics,
    enforce reviewedness threshold, derive lineage, store artifact ZIP,
    and register in the in-memory registry.

    Returns the stored registry entry (ModelOut-like dict).
    """
    import io
    import zipfile

    from src.utils.hf_normalize import normalize_hf_id
    from src.run import compute_metrics_for_model
    from src.services.storage import get_storage
    from src.metrics.treescore import extract_parents_from_resource

    storage = get_storage()

    # Normalize reference into HF ID and URL
    hf_id = normalize_hf_id(model_ref)
    hf_url = f"https://huggingface.co/{hf_id}"

    # ------------------------------------------------------------------
    # Fetch license from Hugging Face (string license, not metric)
    # ------------------------------------------------------------------
    try:
        license_resp = requests.get(f"https://huggingface.co/api/models/{hf_id}")
        hf_meta = license_resp.json()
        hf_license = (hf_meta.get("license") or "").lower()
    except Exception:
        hf_license = ""

    # Prepare scoring request (Phase 1 metrics pipeline)
    base_resource = {
        "name": hf_id,
        "url": hf_url,
        "github_url": None,
        "local_path": None,
        "skip_repo_metrics": False,
        "category": "MODEL",
    }

    # Compute metrics (Phase 1 pipeline output)
    result = compute_metrics_for_model(base_resource)

    # Gate on reviewedness (unchanged behavior)
    reviewedness = result.get("reviewedness", 0.0)
    if reviewedness < 0.5:
        raise HTTPException(
            400, f"Ingest rejected: reviewedness={reviewedness:.2f} < 0.50"
        )

    # ------------------------------------------------------------------
    # Lineage + total_bytes via ScoringService._build_resource
    # ------------------------------------------------------------------
    try:
        enriched_resource = _scoring._build_resource(hf_id)  # type: ignore[attr-defined]
        from src.metrics.treescore import extract_parents_from_resource

        parents = extract_parents_from_resource(enriched_resource)
        total_bytes = enriched_resource.get("total_bytes", 0)
    except Exception:
        parents = []
        total_bytes = 0

    # Inject lineage and size info into metadata
    result["parents"] = parents
    result["total_bytes"] = total_bytes

    # Also store the HF string license in metadata
    result["license"] = hf_license

    # ------------------------------------------------------------------
    # 1. Create registry entry with full metadata (metrics + parents)
    # ------------------------------------------------------------------
    doc = ModelCreate(
        name=hf_id,
        url=hf_url,              # Pydantic will ignore unknown fields, but we pass for clarity
        version="1.0",
        metadata=result,
    )
    created = _registry.create(doc)
    model_id = created["id"]

    # ------------------------------------------------------------------
    # 2. Build a tiny ZIP file for S3/local storage
    # ------------------------------------------------------------------
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("source_url.txt", hf_url)

    zip_bytes = mem_zip.getvalue()

    # ------------------------------------------------------------------
    # 3. Store ZIP blob
    # ------------------------------------------------------------------
    key = f"artifacts/model/{model_id}.zip"
    storage.put_bytes(key, zip_bytes)

    # ------------------------------------------------------------------
    # 4. Generate presigned download URL
    # ------------------------------------------------------------------
    presigned_url = storage.presign(key)

    # ------------------------------------------------------------------
    # 5. Attach download_url to metadata (same dict stored in registry)
    # ------------------------------------------------------------------
    created["metadata"]["download_url"] = presigned_url

    return created


# ---------------------------------------------------------------------------
# Legacy ingest endpoint used by your frontend ("Analyze" button)
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=ModelOut, status_code=201)
def ingest_huggingface(model_ref: str = Query(...)):
    """
    Legacy ingest endpoint used by the frontend.

    - Computes Phase 1 metrics
    - Applies reviewedness gate
    - Stores lineage (parents) and total_bytes in metadata
    - Uploads a tiny ZIP artifact
    - Adds download_url to metadata
    """
    return _ingest_model_from_ref(model_ref)


# ---------------------------------------------------------------------------
# RESET + HEALTH + TRACKS (already mostly spec-compatible)
# ---------------------------------------------------------------------------


@router.delete("/reset", status_code=200)
def reset_system(x_authorization: Optional[str] = Header(default=None, alias="X-Authorization")):
    """
    /reset (BASELINE)

    Spec: requires X-Authorization, but when auth is not implemented the
    header is unused and this should still succeed.
    """
    _registry.reset()
    try:
        _scoring.reset()
    except Exception:
        pass
    return {"status": "registry reset"}


@router.get("/health")
def health():
    """
    /health (BASELINE)

    Spec does not constrain the exact JSON schema, only that 200 means
    service reachable.
    """
    return {
        "status": "ok",
        "uptime_s": int(time.time() - _START_TIME),
        "models": _registry.count_models(),
    }


@router.get("/tracks")
def get_tracks():
    """
    /tracks (spec-compatible)

    Returns the planned tracks for this project.
    """
    return {"plannedTracks": ["Performance track"]}


# ---------------------------------------------------------------------------
# /health/components (NON-BASELINE, simple stub but schema-correct)
# ---------------------------------------------------------------------------


class HealthMetricValue(BaseModel):
    # flexible oneOf; we just treat as Any
    value: Any


from pydantic import RootModel

class HealthMetricMap(RootModel[Dict[str, Any]]):
    pass



class HealthIssue(BaseModel):
    code: str
    severity: str
    summary: str
    details: Optional[str] = None


class HealthLogReference(BaseModel):
    label: str
    url: str
    tail_available: Optional[bool] = None
    last_updated_at: Optional[str] = None


class HealthTimelineEntry(BaseModel):
    bucket: str
    value: float
    unit: Optional[str] = None


class HealthComponentDetail(BaseModel):
    id: str
    display_name: Optional[str] = None
    status: str
    observed_at: str
    description: Optional[str] = None
    metrics: Dict[str, Any] = {}
    issues: List[HealthIssue] = []
    timeline: List[HealthTimelineEntry] = []
    logs: List[HealthLogReference] = []


class HealthComponentCollection(BaseModel):
    components: List[HealthComponentDetail]
    generated_at: str
    window_minutes: int


@router.get("/health/components", response_model=HealthComponentCollection)
def health_components(
    windowMinutes: int = Query(60, ge=5, le=1440),
    includeTimeline: bool = Query(False),
):
    """
    /health/components (NON-BASELINE)

    Minimal but schema-correct implementation.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    comp = HealthComponentDetail(
        id="registry",
        display_name="Registry Service",
        status="ok",
        observed_at=now,
        description="In-memory registry + metrics pipeline.",
        metrics={"model_count": _registry.count_models()},
        issues=[],
        timeline=[] if not includeTimeline else [],
        logs=[],
    )
    return HealthComponentCollection(
        components=[comp],
        generated_at=now,
        window_minutes=windowMinutes,
    )


# ---------------------------------------------------------------------------
# License Check (existing logic, for /models/{id}/license-check)
# ---------------------------------------------------------------------------


class LicenseCheckRequest(BaseModel):
    github_url: str


# Minimal SPDX lookup for compatibility
LICENSE_COMPATIBILITY = {
    "apache-2.0": {"mit", "bsd-3-clause", "bsd-2-clause", "apache-2.0"},
    "mit": {"mit", "bsd-3-clause", "bsd-2-clause"},
    "bsd-3-clause": {"bsd-3-clause", "mit"},
    "bsd-2-clause": {"bsd-2-clause", "mit"},
    "gpl-3.0": set(),
    "cc-by-4.0": set(),
}


def extract_repo_info(url: str):
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            "Invalid GitHub URL. Expected: https://github.com/<owner>/<repo>"
        )
    return parts[0], parts[1]


def fetch_github_license(owner: str, repo: str):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
    resp = requests.get(api_url, headers={"Accept": "application/vnd.github+json"})
    if resp.status_code == 200:
        return resp.json()["license"]["spdx_id"].lower()
    raise ValueError("Unable to determine the GitHub project's license.")


@router.post("/models/{model_id}/license-check")
async def license_check(model_id: str, request: LicenseCheckRequest):
    """
    Existing rich license-check endpoint used by your frontend.

    Returns a structured object with model + GitHub licenses and reason.
    """
    item = _registry.get(model_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    hf_id = item["name"]

    # Get HF license that was stored during ingest
    try:
        model_license = item["metadata"].get("license", "")
        if not model_license:
            raise HTTPException(
                status_code=500, detail="Model has no license stored. Re-ingest the model."
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch model license: {e}"
        )

    # GitHub license
    try:
        owner, repo = extract_repo_info(request.github_url)
        github_license = fetch_github_license(owner, repo)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    compatible = github_license in LICENSE_COMPATIBILITY.get(model_license, set())
    reason = (
        f"{github_license.upper()} is compatible with {model_license.upper()}."
        if compatible
        else f"{github_license.upper()} is NOT compatible with {model_license.upper()} "
        f"for fine-tune + inference."
    )

    return {
        "model_id": model_id,
        "model_name": hf_id,
        "model_license": model_license,
        "github_license": github_license,
        "compatible": compatible,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Lineage (existing tree endpoint for frontend)
# ---------------------------------------------------------------------------


@router.get("/models/{model_id}/lineage")
def get_lineage(model_id: str):
    """
    Existing lineage endpoint used by your frontend. Returns your
    custom {root_id, nodes, edges} JSON.
    """
    try:
        return _registry.get_lineage_graph(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")


# ---------------------------------------------------------------------------
# PHASE 2 SPEC: Artifact Data & Metadata models (for typing / docs)
# ---------------------------------------------------------------------------


class ArtifactData(BaseModel):
    url: str
    download_url: Optional[str] = None


class ArtifactMetadata(BaseModel):
    name: str
    id: str
    type: str  # "model" | "dataset" | "code"


class Artifact(BaseModel):
    metadata: ArtifactMetadata
    data: ArtifactData


class ArtifactQuery(BaseModel):
    name: str
    types: Optional[List[str]] = None


class ArtifactRegEx(BaseModel):
    regex: str


# Small utility to convert our internal registry entry into spec Artifact
def _entry_to_artifact(entry: Dict[str, Any]) -> Dict[str, Any]:
    meta = entry.get("metadata", {}) or {}
    source_url = meta.get("url") or meta.get("source_uri")
    download_url = meta.get("download_url")

    return {
        "metadata": {
            "name": entry["name"],
            "id": entry["id"],
            "type": "model",  # we only support models for now
        },
        "data": {
            "url": source_url or f"https://huggingface.co/{entry['name']}",
            "download_url": download_url,
        },
    }


# ---------------------------------------------------------------------------
# PHASE 2 BASELINE: /artifact/{artifact_type}  (Create artifact)
# ---------------------------------------------------------------------------


@router.post(
    "/artifact/{artifact_type}",
    status_code=201,
    response_model=Artifact,
)
def artifact_create(
    artifact_type: str,
    body: ArtifactData,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifact/{artifact_type}  (BASELINE)

    We support artifact_type="model" and ingest from Hugging Face URLs/ids
    using the same pipeline as /ingest.
    """
    if artifact_type != "model":
        raise HTTPException(
            status_code=400, detail="Only artifact_type='model' is supported."
        )

    created = _ingest_model_from_ref(body.url)
    return _entry_to_artifact(created)


# ---------------------------------------------------------------------------
# PHASE 2 BASELINE: /artifacts  (List / search artifacts)
# ---------------------------------------------------------------------------


@router.post("/artifacts", response_model=List[ArtifactMetadata])
def artifacts_list(
    queries: List[ArtifactQuery],
    response: Response,
    offset: Optional[str] = Query(
        default=None,
        description="Pagination offset; if absent, start at first page.",
    ),
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifacts (BASELINE)

    Use:
      - name="*" to enumerate all artifacts.
      - name="foo" to filter by exact name.
      - types ignored except that we only support 'model'.
    """
    if not queries:
        raise HTTPException(
            status_code=400,
            detail="At least one ArtifactQuery is required.",
        )

    q = queries[0]
    name = q.name
    types = q.types or []

    # We only support models
    if types and "model" not in types:
        return []

    # Interpret offset as index into filtered list
    try:
        start = int(offset) if offset is not None else 0
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid offset value.")

    all_entries = list(_registry._models)  # internal list is ordered

    if name == "*":
        filtered = all_entries
    else:
        filtered = [m for m in all_entries if m.get("name") == name]

    page_size = 50
    page = filtered[start : start + page_size]
    next_offset = start + len(page) if (start + len(page)) < len(filtered) else None

    # Set offset header for pagination
    if next_offset is not None:
        response.headers["offset"] = str(next_offset)
    else:
        response.headers["offset"] = ""

    return [
        ArtifactMetadata(name=e["name"], id=e["id"], type="model") for e in page
    ]


# ---------------------------------------------------------------------------
# PHASE 2 BASELINE: /artifact/{artifact_type}/{id} (GET/PUT/DELETE)
# ---------------------------------------------------------------------------


@router.get("/artifact/{artifact_type}/{id}", response_model=Artifact)
def artifact_get(
    artifact_type: str,
    id: str,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    if artifact_type != "model":
        raise HTTPException(status_code=400, detail="Only 'model' artifacts supported.")

    entry = _registry.get(id)
    if not entry:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    return _entry_to_artifact(entry)


@router.put("/artifact/{artifact_type}/{id}")
def artifact_update(
    artifact_type: str,
    id: str,
    body: Artifact,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    if artifact_type != "model":
        raise HTTPException(status_code=400, detail="Only 'model' artifacts supported.")

    entry = _registry.get(id)
    if not entry:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    # Spec: name and id must match
    if body.metadata.id != id or body.metadata.name != entry["name"]:
        raise HTTPException(
            status_code=400,
            detail="Artifact name/id mismatch.",
        )

    # For now we just update the stored source URL / download_url in metadata
    meta = entry.setdefault("metadata", {})
    if body.data.url:
        meta["url"] = body.data.url
    if body.data.download_url:
        meta["download_url"] = body.data.download_url

    return {"status": "updated"}


@router.delete("/artifact/{artifact_type}/{id}")
def artifact_delete(
    artifact_type: str,
    id: str,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    if artifact_type != "model":
        raise HTTPException(status_code=400, detail="Only 'model' artifacts supported.")

    ok = _registry.delete(id)
    if not ok:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# PHASE 2 BASELINE: /artifact/model/{id}/rate   (ModelRating)
# ---------------------------------------------------------------------------


@router.get("/artifact/model/{id}")
def artifact_model_get_stub(
    id: str,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    NOTE: The spec does NOT define /artifact/model/{id} itself as a rating
    endpoint; rating is /artifact/model/{id}/rate. This stub is here only
    if something accidentally calls it; we just proxy to /artifact/model/{id}/rate.
    """
    # For simplicity, just delegate to /artifact/model/{id}/rate
    return artifact_model_rate(id, x_authorization)


@router.get("/artifact/model/{id}/rate")
def artifact_model_rate(
    id: str,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifact/model/{id}/rate (BASELINE)

    Build a ModelRating-shaped JSON object from the metrics stored in
    registry metadata (populated during ingest via compute_metrics_for_model).
    """
    entry = _registry.get(id)
    if not entry:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    md = entry.get("metadata", {}) or {}
    name = md.get("name") or entry.get("name") or ""
    category = md.get("category") or "model"

    def ms_to_sec(v):
        try:
            return float(v) / 1000.0
        except Exception:
            try:
                return float(v)
            except Exception:
                return 0.0

    # size_score in spec is a map; in Phase 1 it was "size"
    size_map = md.get("size_score") or md.get("size") or {}
    return {
        "name": name,
        "category": category,
        "net_score": md.get("net_score", 0.0),
        "net_score_latency": ms_to_sec(md.get("net_score_latency", 0)),
        "ramp_up_time": md.get("ramp_up_time", 0.0),
        "ramp_up_time_latency": ms_to_sec(md.get("ramp_up_time_latency", 0)),
        "bus_factor": md.get("bus_factor", 0.0),
        "bus_factor_latency": ms_to_sec(md.get("bus_factor_latency", 0)),
        "performance_claims": md.get("performance_claims", 0.0),
        "performance_claims_latency": ms_to_sec(
            md.get("performance_claims_latency", 0)
        ),
        "license": md.get("license_score", md.get("license", 0.0))
        if isinstance(md.get("license_score", md.get("license", 0.0)), (int, float))
        else 0.0,
        "license_latency": ms_to_sec(md.get("license_latency", 0)),
        "dataset_and_code_score": md.get("dataset_and_code_score", 0.0),
        "dataset_and_code_score_latency": ms_to_sec(
            md.get("dataset_and_code_score_latency", 0)
        ),
        "dataset_quality": md.get("dataset_quality", 0.0),
        "dataset_quality_latency": ms_to_sec(md.get("dataset_quality_latency", 0)),
        "code_quality": md.get("code_quality", 0.0),
        "code_quality_latency": ms_to_sec(md.get("code_quality_latency", 0)),
        "reproducibility": md.get("reproducibility", 0.0),
        "reproducibility_latency": ms_to_sec(md.get("reproducibility_latency", 0)),
        "reviewedness": md.get("reviewedness", 0.0),
        "reviewedness_latency": ms_to_sec(md.get("reviewedness_latency", 0)),
        "tree_score": md.get("tree_score", md.get("treescore", 0.0)),
        "tree_score_latency": ms_to_sec(
            md.get("tree_score_latency", md.get("treescore_latency", 0))
        ),
        "size_score": {
            "raspberry_pi": size_map.get("raspberry_pi", 0.0),
            "jetson_nano": size_map.get("jetson_nano", 0.0),
            "desktop_pc": size_map.get("desktop_pc", 0.0),
            "aws_server": size_map.get("aws_server", 0.0),
        },
        "size_score_latency": ms_to_sec(md.get("size_latency", md.get("size_score_latency", 0))),
    }


# ---------------------------------------------------------------------------
# PHASE 2 BASELINE: /artifact/model/{id}/license-check (boolean)
# ---------------------------------------------------------------------------


@router.post("/artifact/model/{id}/license-check")
def artifact_model_license_check(
    id: str,
    request: LicenseCheckRequest,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifact/model/{id}/license-check (BASELINE)

    Reuses the richer /models/{model_id}/license-check logic, but returns
    only the boolean compatibility flag as required by the spec.
    """
    # Reuse existing logic
    result = license_check(id, request)
    # license_check can raise HTTPException; we propagate those
    return bool(result["compatible"])


# ---------------------------------------------------------------------------
# PHASE 2 BASELINE: /artifact/model/{id}/lineage
# ---------------------------------------------------------------------------


@router.get("/artifact/model/{id}/lineage")
def artifact_model_lineage(
    id: str,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifact/model/{id}/lineage (BASELINE)

    Returns an ArtifactLineageGraph-shaped object. We map your internal
    graph (if any) to the spec schema. If no parents/children are present,
    returns a graph with a single node and no edges.
    """
    try:
        internal = _registry.get_lineage_graph(id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    # internal: { root_id, nodes: [{id, name}], edges: [{parent, child}] }
    id_to_node = {n["id"]: n for n in internal.get("nodes", [])}

    # Build nodes array
    nodes = []
    for nid, n in id_to_node.items():
        nodes.append(
            {
                "artifact_id": nid,
                "name": n.get("name", ""),
                "source": "registry",
                "metadata": {},
            }
        )

    # Build edges array
    edges = []
    for e in internal.get("edges", []):
        edges.append(
            {
                "from_node_artifact_id": e["parent"],
                "to_node_artifact_id": e["child"],
                "relationship": "base_model",  # simple label
            }
        )

    # If for some reason there are no nodes, still return a single node for id
    if not nodes:
        entry = _registry.get(id)
        if not entry:
            raise HTTPException(status_code=404, detail="Artifact does not exist.")
        nodes.append(
            {
                "artifact_id": id,
                "name": entry["name"],
                "source": "registry",
                "metadata": {},
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# PHASE 2 BASELINE: /artifact/{artifact_type}/{id}/cost
# ---------------------------------------------------------------------------


@router.get("/artifact/{artifact_type}/{id}/cost")
def artifact_cost(
    artifact_type: str,
    id: str,
    dependency: bool = Query(False),
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifact/{artifact_type}/{id}/cost (BASELINE)

    We approximate "cost" as total download size in MB using the
    total_bytes stored in metadata during ingest. For now, dependencies
    are not expanded; standalone_cost == total_cost.
    """
    if artifact_type != "model":
        raise HTTPException(status_code=400, detail="Only 'model' artifacts supported.")

    entry = _registry.get(id)
    if not entry:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    md = entry.get("metadata", {}) or {}
    total_bytes = md.get("total_bytes", 0)
    try:
        mb = float(total_bytes) / (1024.0 * 1024.0)
    except Exception:
        mb = 0.0

    if not dependency:
        return {
            id: {
                "total_cost": mb,
            }
        }

    # With dependency=true, we must include standalone_cost + total_cost
    return {
        id: {
            "standalone_cost": mb,
            "total_cost": mb,
        }
    }


# ---------------------------------------------------------------------------
# PHASE 2 NON-BASELINE: /artifact/byRegEx
# ---------------------------------------------------------------------------


@router.post("/artifact/byRegEx", response_model=List[ArtifactMetadata])
def artifact_by_regex(
    body: ArtifactRegEx,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifact/byRegEx (BASELINE)

    Search over artifact names and READMEs using a regular expression,
    using the same semantics as your original /models?q=regex behavior.
    """
    import re

    pattern = body.regex
    try:
        pat = re.compile(pattern)
    except re.error:
        raise HTTPException(status_code=400, detail="Invalid regular expression.")

    matches: List[ArtifactMetadata] = []
    for e in _registry._models:
        name = e.get("name", "")
        card = str(e.get("metadata", {}).get("card", ""))
        if pat.search(name) or pat.search(card):
            matches.append(
                ArtifactMetadata(name=e["name"], id=e["id"], type="model")
            )

    if not matches:
        raise HTTPException(status_code=404, detail="No artifact found under this regex.")

    return matches


# ---------------------------------------------------------------------------
# PHASE 2 NON-BASELINE: /artifact/byName/{name}
# ---------------------------------------------------------------------------


@router.get("/artifact/byName/{name}", response_model=List[ArtifactMetadata])
def artifact_by_name(
    name: str,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifact/byName/{name} (NON-BASELINE)

    Simple exact-name match over the registry.
    """
    results: List[ArtifactMetadata] = []
    for e in _registry._models:
        if e.get("name") == name:
            results.append(
                ArtifactMetadata(name=e["name"], id=e["id"], type="model")
            )

    if not results:
        raise HTTPException(status_code=404, detail="No such artifact.")

    return results


# ---------------------------------------------------------------------------
# PHASE 2 NON-BASELINE: /artifact/{artifact_type}/{id}/audit
# ---------------------------------------------------------------------------


@router.get("/artifact/{artifact_type}/{id}/audit")
def artifact_audit(
    artifact_type: str,
    id: str,
    x_authorization: Optional[str] = Header(default=None, alias="X-Authorization"),
):
    """
    /artifact/{artifact_type}/{id}/audit (NON-BASELINE)

    Minimal implementation: we do not keep a full audit log, so we
    return an empty list. Shape matches the spec's array of
    ArtifactAuditEntry.
    """
    if artifact_type != "model":
        raise HTTPException(status_code=400, detail="Only 'model' artifacts supported.")

    entry = _registry.get(id)
    if not entry:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    # Return empty audit trail
    return []


# ---------------------------------------------------------------------------
# PHASE 2: /authenticate (NOT IMPLEMENTED -> 501)
# ---------------------------------------------------------------------------


class AuthenticationRequest(BaseModel):
    user: Dict[str, Any]
    secret: Dict[str, Any]


@router.put("/authenticate")
def authenticate(body: AuthenticationRequest):
    """
    /authenticate

    Per spec: if your system does NOT implement auth, this endpoint
    should return 501 and X-Authorization is unused everywhere else.
    """
    raise HTTPException(
        status_code=501, detail="Authentication is not implemented in this system."
    )
