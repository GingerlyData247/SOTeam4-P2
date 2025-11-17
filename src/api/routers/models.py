from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import time

from ...schemas.models import ModelCreate, ModelUpdate, ModelOut, Page
from ...services.registry import RegistryService
from ...services.ingest import IngestService
from ...services.scoring import ScoringService

_START_TIME = time.time()

router = APIRouter()

_registry = RegistryService()
_ingest = IngestService(registry=_registry)
_scoring = ScoringService()


# ------------------------------------------------------------------ #
# CRUD endpoints
# ------------------------------------------------------------------ #
@router.post("/models", response_model=ModelOut, status_code=201)
def create_model(body: ModelCreate) -> ModelOut:
    return _registry.create(body)


@router.get("/models", response_model=Page[ModelOut])
def list_models(
    q: Optional[str] = Query(default=None, description="Regex over name/card"),
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = None,
) -> Page[ModelOut]:
    return _registry.list(q=q, limit=limit, cursor=cursor)


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


# ------------------------------------------------------------------ #
# Rating & Ingest
# ------------------------------------------------------------------ #
@router.get("/rate/{model_ref:path}")
def rate_model(model_ref: str):
    """
    model_ref can be 'owner/name' or a local id; we accept both for Dev UX.
    Returns: {"net": float, "subs": {...}, "latency_ms": int}
    """
    import io, os, time, shutil
    from contextlib import redirect_stdout, redirect_stderr
    from dotenv import load_dotenv
    from src.utils.hf_normalize import normalize_hf_id
    from src.utils.github_link_finder import find_github_url_from_hf
    from src.utils.repo_cloner import clone_repo_to_temp
    from run import compute_metrics_for_model, _normalize_github_repo_url

    load_dotenv()
    start = time.perf_counter()

    # 1️⃣ Normalize model name and build HF URL
    hf_id = normalize_hf_id(model_ref)
    hf_url = f"https://huggingface.co/{hf_id}"

    # 2️⃣ Try to find a corresponding GitHub repo
    # Phase 2: No GitHub cloning; metrics must run without repos
    repo_url = None
    local_path = None


    # 4️⃣ Build resource (same as CLI)
    resource = {
    "name": hf_id,
    "url": hf_url,
    "github_url": None,
    "local_path": None,
    "skip_repo_metrics": False,
    "category": "MODEL",
    }


    # 5️⃣ Run metric computation
    result = compute_metrics_for_model(resource)

    # 6️⃣ Clean up cloned repo
    if local_path:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            shutil.rmtree(local_path, ignore_errors=True)

    latency_ms = int((time.perf_counter() - start) * 1000)
    subs = {k: v for k, v in result.items() if isinstance(v, (float, dict)) and not k.endswith("_latency")}

    return {
        "net": result.get("net_score", 0.0),
        "subs": subs,
        "latency_ms": latency_ms
    }




@router.post("/ingest", response_model=ModelOut, status_code=201)
def ingest_huggingface(model_ref: str = Query(..., description="owner/name or full HF URL")) -> ModelOut:
    """
    Ingest a Hugging Face model, ensuring metrics (like reviewedness) are computed
    with the same repo-cloning logic as /rate.
    """
    import io, shutil
    from contextlib import redirect_stdout, redirect_stderr
    from src.utils.hf_normalize import normalize_hf_id
    from src.utils.github_link_finder import find_github_url_from_hf
    from src.utils.repo_cloner import clone_repo_to_temp
    from run import compute_metrics_for_model, _normalize_github_repo_url
    from ...schemas.models import ModelCreate

    hf_id = normalize_hf_id(model_ref)
    hf_url = f"https://huggingface.co/{hf_id}"
    # Phase 2: No GitHub cloning; metrics must run without repos
    repo_url = None
    local_path = None


    resource = {
    "name": hf_id,
    "url": hf_url,
    "github_url": None,
    "local_path": None,
    "skip_repo_metrics": False,
    "category": "MODEL",
    }

    result = compute_metrics_for_model(resource)

    # Cleanup
    if local_path:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            shutil.rmtree(local_path, ignore_errors=True)

    reviewedness = result.get("reviewedness", 0.0)
    if reviewedness < 0.5:
        raise HTTPException(status_code=400, detail=f"Ingest rejected: reviewedness={reviewedness:.2f} < 0.50")

    # ✅ Create proper Pydantic model for registry
    model_create = ModelCreate(
        name=hf_id,
        url=hf_url,
        version="1.0",
        metadata=result
    )

    model_entry = _registry.create(model_create)
    return model_entry


# ------------------------------------------------------------------ #
# Reset & Health
# ------------------------------------------------------------------ #
@router.post("/reset", status_code=204)
def reset_system():
    _registry.reset()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_s": int(time.time() - _START_TIME),
        "models": _registry.count_models(),
    }
