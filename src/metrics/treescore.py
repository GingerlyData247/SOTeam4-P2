# src/metrics/treescore.py
from __future__ import annotations
import time, importlib
from typing import Any, Dict, Tuple, Callable, List, Set

from huggingface_hub import HfApi, ModelInfo
from src.utils.hf_normalize import normalize_hf_id
from src.utils.logging import logger

api = HfApi()

_INFO: Dict[str, ModelInfo] = {}
_NET_CACHE: Dict[str, float] = {}
_PARENT_CACHE: Dict[str, List[str]] = {}
_METRICS: Dict[str, Callable] = {}

OTHER = [
    "bus_factor",
    "category",
    "code_quality",
    "dataset_and_code_score",
    "dataset_quality",
    "license",
    "performance_claims",
    "ramp_up_time",
    "reproducibility",
    "reviewedness",
    "size",
]

def _info(repo: str) -> ModelInfo | None:
    repo = normalize_hf_id(repo)
    if repo in _INFO:
        return _INFO[repo]
    try:
        m = api.model_info(repo)
        _INFO[repo] = m
        return m
    except Exception as e:
        logger.debug("treescore: info failed for %s: %s", repo, e)
        return None

def _load_metrics():
    global _METRICS
    if _METRICS:
        return _METRICS
    for name in OTHER:
        try:
            mod = importlib.import_module(f"src.metrics.{name}")
            fn = getattr(mod, "metric", None)
            if callable(fn):
                _METRICS[name] = fn
        except Exception as e:
            logger.debug("treescore: skip metric %s: %s", name, e)
    return _METRICS

def _scalar(name: str, val: Any) -> float | None:
    if isinstance(val, (int, float)):
        return float(max(0, min(1, val)))
    if name == "size" and isinstance(val, dict):
        nums = [v for v in val.values() if isinstance(v, (int, float))]
        if nums:
            avg = sum(nums)/len(nums)
            return float(max(0, min(1, avg)))
    return None

def _net(repo: str) -> float:
    repo = normalize_hf_id(repo)
    if repo in _NET_CACHE:
        return _NET_CACHE[repo]

    m = _info(repo)
    if m is None:
        _NET_CACHE[repo] = 0.0
        return 0.0

    metrics = _load_metrics()
    resource = {
        "name": repo,
        "url": f"https://huggingface.co/{repo}",
        "category": "MODEL",
        "skip_repo_metrics": False,
        "local_path": None,
    }

    scores = []
    for name, fn in metrics.items():
        try:
            s, _lat = fn(resource)
            sc = _scalar(name, s)
            if sc is not None:
                scores.append(sc)
        except:
            pass

    net = float(sum(scores)/len(scores)) if scores else 0.0
    _NET_CACHE[repo] = net
    return net

def _parents(repo: str) -> List[str]:
    repo = normalize_hf_id(repo)
    if repo in _PARENT_CACHE:
        return _PARENT_CACHE[repo]

    info = _info(repo)
    if info is None:
        _PARENT_CACHE[repo] = []
        return []

    pts = set()
    card_obj = getattr(info, "cardData", None)
    card = getattr(card_obj, "data", {}) if card_obj else {}
    tags = [t for t in getattr(info, "tags", []) if isinstance(t, str)]

    for key in ("base_model", "parent_model", "source_model", "teacher_model", "original_model"):
        v = card.get(key)
        if isinstance(v, str):
            pts.add(normalize_hf_id(v))

    for t in tags:
        low = t.lower()
        if low.startswith(("base_model:", "parent_model:")):
            val = t.split(":", 1)[1].strip()
            if val:
                pts.add(normalize_hf_id(val))

    for t in tags:
        if "/" in t and not t.startswith(("task:", "pipeline:", "license:", "arxiv:")):
            pts.add(normalize_hf_id(t))

    plist = list(pts)
    _PARENT_CACHE[repo] = plist
    return plist

def _walk(repo: str, seen: Set[str]) -> Tuple[float, int]:
    repo = normalize_hf_id(repo)
    if repo in seen:
        return 0.0, 0
    seen.add(repo)

    parents = _parents(repo)
    total = 0.0
    count = 0

    for p in parents:
        pn = _net(p)
        total += pn
        count += 1

        ssum, scnt = _walk(p, seen)
        total += ssum
        count += scnt

    return total, count
    
def extract_parents_from_resource(resource: Dict[str, Any]) -> List[str]:
    """
    Derive a list of *candidate parent models* for a given model resource.

    This is used by the API's lineage graph endpoint.

    Differences from `_parents(repo)` above:
    - `_parents(repo)` talks to the Hugging Face API (HfApi) using a model id
      string like "facebook/wav2vec2-base".
    - `extract_parents_from_resource(resource)` is "offline": it only inspects
      fields already present in the `resource` dict (config.json, tags, etc.),
      which keeps lineage reproducible and avoids extra network calls.

    The Phase 2 spec says lineage must be derived from structured metadata
    (config.json), so we prioritize that here.
    """
    parents: Set[str] = set()

    # Normalized name of *this* model, so we can avoid listing itself as a parent
    name = resource.get("name") or ""
    if isinstance(name, str) and "/" in name:
        name = normalize_hf_id(name)

    # -------------------------------------------------------------------------
    # 1) PRIMARY SOURCE: config.json (structured metadata)
    #    We expect prepare_resource / ingest to have loaded config.json into
    #    resource["config"] as a dict. Different HF models use different keys
    #    for their "base model", so we check several common ones.
    # -------------------------------------------------------------------------
    cfg = resource.get("config") or {}
    if isinstance(cfg, dict):
        candidate_keys = (
            "base_model",
            "teacher_model",
            "parent_model",
            "source_model",
            "original_model",
            "pretrained_model_name_or_path",
        )
        for key in candidate_keys:
            val = cfg.get(key)
            if isinstance(val, str) and "/" in val:
                parents.add(normalize_hf_id(val))

    # -------------------------------------------------------------------------
    # 2) SECONDARY SOURCE: HF metadata tags (if present in the resource)
    #    If prepare_resource stored Hugging Face metadata under
    #    resource["hf_metadata"]["tags"], we can re-use those. We filter out
    #    non-model tags like "task:", "pipeline:", "license:", etc.
    # -------------------------------------------------------------------------
    hf_meta = resource.get("hf_metadata") or {}
    if isinstance(hf_meta, dict):
        tags = hf_meta.get("tags") or []
        for t in tags:
            if not isinstance(t, str):
                continue
            if "/" not in t:
                continue
            if t.startswith(("task:", "pipeline:", "license:", "arxiv:")):
                continue
            parents.add(normalize_hf_id(t))

    # Do not list the model itself as its own parent, even if metadata is messy
    parents.discard(name)

    # Sorted list is nicer for determinism and testing
    return sorted(parents)

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    start = time.perf_counter()
    name = resource.get("name") or ""
    if "/" not in name:
        return 0.0, int((time.perf_counter() - start)*1000)

    repo = normalize_hf_id(name)

    try:
        own = _net(repo)
        s, c = _walk(repo, seen=set())

        if c == 0:
            score = own
        else:
            avg = s/c
            score = (own + avg) / 2.0

        score = float(max(0, min(1, score)))
    except Exception as e:
        logger.debug("treescore: failed for %s: %s", repo, e)
        score = 0.0

    lat = int((time.perf_counter() - start)*1000)
    return score, lat