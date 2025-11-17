from fastapi import APIRouter, HTTPException, Query
from typing import Optional
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
# Autograder-Required Reset Endpoints
# ------------------------------------------------------------------ #
@router.delete("/artifacts/reset", status_code=200)
def reset_artifacts():
    _registry.reset()
    try:
        _scoring.reset()
    except Exception:
        pass
    return {"status": "registry reset"}


@router.delete("/models/reset", status_code=200)
def reset_models():
    _registry.reset()
    try:
        _scoring.reset()
    except Exception:
        pass
    return {"status": "registry reset"}


# ------------------------------------------------------------------ #
# CRUD Endpoints
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
# Rating
# ------------------------------------------------------------------ #
@router.get("/rate/{model_ref:path}")
def rate_model(model_ref: str):
    import io, shutil, time as _time
    from contextlib import redirect_stdout, redirect_stderr
    from dotenv import load_dotenv
    from src.utils.hf_normalize import normalize_hf_id
    from src.run import compute_metrics_for_model

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

    subs = {
        k: v
        for k, v in result.items()
        if not k.endswith("_latency") and isinstance(v, (float, dict))
    }

    return {
        "net": result.get("net_score", 0.0),
        "subs": subs,
        "latency_ms": latency_ms,
    }


# ------------------------------------------------------------------ #
# Ingest
# ------------------------------------------------------------------ #
@router.post("/ingest", response_model=ModelOut, status_code=201)
def ingest_huggingface(
    model_ref: str = Query(..., description="owner/name or full HF URL"),
) -> ModelOut:

    from src.utils.hf_normalize import normalize_hf_id
    from src.run import compute_metrics_for_model
    from ...schemas.models import ModelCreate

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

    reviewedness = result.get("reviewedness", 0.0)
    if reviewedness < 0.5:
        raise HTTPException(
            status_code=400,
            detail=f"Ingest rejected: reviewedness={reviewedness:.2f} < 0.50"
        )

    model_create = ModelCreate(
        name=hf_id,
        url=hf_url,
        version="1.0",
        metadata=result,
    )

    return _registry.create(model_create)


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #
@router.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_s": int(time.time() - _START_TIME),
        "models": _registry.count_models(),
    }


# ------------------------------------------------------------------ #
# Tracks
# ------------------------------------------------------------------ #
@router.get("/tracks")
def get_tracks():
    return {"plannedTracks": ["Performance track"]}
