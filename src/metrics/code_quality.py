# src/metrics/code_quality.py
from __future__ import annotations
import time
from typing import Any, Dict, Tuple, List

from huggingface_hub import HfApi, ModelInfo
from src.utils.hf_normalize import normalize_hf_id
from src.utils.logging import logger

api = HfApi()
_INFO_CACHE: Dict[str, ModelInfo] = {}

def _get_info(resource: Dict[str, Any]) -> ModelInfo | None:
    name = resource.get("name") or ""
    if "/" not in name: 
        return None

    repo = normalize_hf_id(name)
    if repo in _INFO_CACHE:
        return _INFO_CACHE[repo]

    try:
        info = api.model_info(repo)
        _INFO_CACHE[repo] = info
        return info
    except Exception as e:
        logger.debug("code_quality: model_info failed for %s: %s", repo, e)
        return None


def _siblings(info: ModelInfo) -> List[Any]:
    return getattr(info, "siblings", None) or []


def _has_readme(files: List[Any]) -> bool:
    for f in files:
        try:
            if f.rfilename.lower().startswith("readme"):
                return True
        except:
            pass
    return False


def _has_examples_or_ipynb(files: List[Any]) -> bool:
    for f in files:
        try:
            p = f.rfilename.lower()
            if "/examples/" in p or p.endswith(".ipynb"):
                return True
        except:
            pass
    return False


def _has_config(files: List[Any]) -> bool:
    cfgs = {
        "config.json", "preprocessor_config.json", "model_index.json",
        "tokenizer.json", "pyproject.toml", "setup.py", "requirements.txt"
    }
    for f in files:
        try:
            name = f.rfilename.lower()
            if name in cfgs or any(name.endswith(c) for c in cfgs):
                return True
        except:
            pass
    return False


def _doc_score(info: ModelInfo) -> float:
    files = _siblings(info)
    card_obj = getattr(info, "cardData", None)
    card = getattr(card_obj, "data", {}) if card_obj else {}

    score = 0.0
    if _has_readme(files):
        score += 0.4

    if card:
        score += 0.3
        if any(k in card for k in ("usage", "model-index", "language", "datasets", "metrics")):
            score += 0.2

    if _has_examples_or_ipynb(files):
        score += 0.1

    return min(1.0, score)


def _structure_score(info: ModelInfo) -> float:
    files = _siblings(info)
    score = 0.2
    if _has_config(files):
        score += 0.3
    if _has_examples_or_ipynb(files):
        score += 0.3
    if len(files) >= 10:
        score += 0.2
    return min(1.0, score)


def _popularity(info: ModelInfo) -> float:
    d = float(getattr(info, "downloads", 0) or 0)
    if d >= 10_000_000: return 1.0
    if d >= 1_000_000:  return 0.9
    if d >= 100_000:   return 0.75
    if d >= 10_000:    return 0.6
    if d >= 1_000:     return 0.4
    if d > 0:          return 0.2
    return 0.0


def _library_score(info: ModelInfo) -> float:
    tags = [t.lower() for t in getattr(info, "tags", [])]
    library = (getattr(info, "library_name", "") or "").lower()
    pipe = (getattr(info, "pipeline_tag", "") or "").lower()

    score = 0.3
    if "transformers" in tags or library == "transformers":
        score = 1.0
    elif library in {"pytorch", "tensorflow", "keras", "sklearn"}:
        score = 0.8
    if pipe:
        score = max(score, 0.7)
    return min(1.0, score)


def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    start = time.perf_counter()
    info = _get_info(resource)
    if info is None:
        return 0.0, int((time.perf_counter() - start) * 1000)

    doc = _doc_score(info)
    struct = _structure_score(info)
    pop = _popularity(info)
    lib = _library_score(info)

    score = (
        0.35 * doc +
        0.25 * struct +
        0.20 * pop +
        0.20 * lib
    )
    score = float(max(0, min(1, score)))
    return score, int((time.perf_counter() - start) * 1000)
