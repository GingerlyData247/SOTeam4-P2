# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# METRIC: reviewedness
# REQUIREMENTS SATISFIED: reviewedness metric score

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/metrics/reviewedness.py

Computes the Reviewedness metric for a model artifact.

This metric estimates how thoroughly a model has been reviewed and
validated by the community by combining multiple signals from the
Hugging Face ecosystem. These signals include model popularity,
community engagement, and the presence of structured evaluation or
documentation in the model card.

Scoring factors:
    - Download count as a proxy for adoption and exposure
    - Like count as a measure of community approval
    - Model card content indicating evaluation results, datasets,
      licenses, or academic references

The metric follows the Phase 2 specification and is designed to be
fault-tolerant, always returning a bounded score and measured latency
even when metadata retrieval fails.
"""
# src/metrics/reviewedness.py
from __future__ import annotations
import time
from typing import Any, Dict, Tuple

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
        logger.debug("reviewedness: model_info failed for %s: %s", repo, e)
        return None

def _downloads(info: ModelInfo) -> float:
    d = float(getattr(info, "downloads", 0) or 0)
    if d >= 20_000_000: return 1.0
    if d >= 5_000_000:  return 0.9
    if d >= 1_000_000:  return 0.8
    if d >= 100_000:    return 0.6
    if d >= 10_000:     return 0.4
    if d >= 1_000:      return 0.2
    if d > 0:           return 0.1
    return 0.0

def _likes(info: ModelInfo) -> float:
    l = int(getattr(info, "likes", 0) or 0)
    if l >= 1000: return 1.0
    if l >= 200:  return 0.8
    if l >= 50:   return 0.6
    if l >= 10:   return 0.4
    if l >= 1:    return 0.2
    return 0.0

def _card(info: ModelInfo) -> float:
    card_obj = getattr(info, "cardData", None)
    card = getattr(card_obj, "data", {}) if card_obj else {}

    if not isinstance(card, dict):
        return 0.0

    score = 0.3
    if any(k in card for k in ("model-index", "metrics", "evaluation", "results")):
        score += 0.4
    if any(k in card for k in ("datasets", "language", "license")):
        score += 0.2

    # detect arxiv links inside card values
    try:
        if any("arxiv" in str(v).lower() for v in card.values()):
            score += 0.1
    except:
        pass

    return min(1.0, score)

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    start = time.perf_counter()
    info = _get_info(resource)
    if info is None:
        return 0.0, int((time.perf_counter() - start) * 1000)

    d = _downloads(info)
    l = _likes(info)
    c = _card(info)

    score = 0.60*d + 0.25*l + 0.15*c
    score = float(max(0, min(1, score)))
    return score, int((time.perf_counter() - start) * 1000)
