# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# METRIC: license
# REQUIREMENTS SATISFIED: license metric score

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/metrics/license.py

Computes the License Compatibility metric for a model artifact.

This metric evaluates the suitability of a model’s license for reuse by
examining license metadata provided by Hugging Face. It assigns higher
scores to permissive open-source licenses and lower scores to weak
copyleft, non-commercial, or uncommon license types.

Scoring categories:
    - 1.0 for clearly permissive licenses (Apache-2.0, MIT, BSD, etc.)
    - 0.6 for weak copyleft or non-commercial licenses
    - 0.3 for uncommon or unrecognized license strings
    - 0.0 when no license information is available

The metric is Hugging Face–only, requires no external repository access,
and degrades safely when license metadata cannot be retrieved, always
returning a bounded score and measured latency.
"""
# src/metrics/license.py
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from huggingface_hub import HfApi, ModelInfo
from src.utils.hf_normalize import normalize_hf_id
from src.utils.logging import logger


api = HfApi()
_HF_INFO_CACHE: Dict[str, ModelInfo] = {}


def _get_model_info(resource: Dict[str, Any]) -> ModelInfo | None:
    """
    Best-effort fetch of Hugging Face model info.
    Returns None if this is not an HF model or the API call fails.
    """
    name = resource.get("name") or ""
    if "/" not in name:
        return None

    repo_id = normalize_hf_id(name)
    if repo_id in _HF_INFO_CACHE:
        return _HF_INFO_CACHE[repo_id]

    try:
        info = api.model_info(repo_id)
        _HF_INFO_CACHE[repo_id] = info
        return info
    except Exception as e:
        logger.debug("license.metric: failed to fetch model_info for %s: %s", repo_id, e)
        return None


def _license_score(lic: str | None) -> float:
    if not lic:
        return 0.0

    l = lic.lower().strip()

    # Strongly permissive licenses → full score
    permissive = {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "mpl-2.0",
        "unlicense",
        "cc-by-4.0",
        "cc-by-sa-4.0",
    }

    # Somewhat permissive / weak copyleft / non-commercial → partial credit
    weak = {
        "lgpl-2.1",
        "lgpl-3.0",
        "epl-2.0",
        "cc-by-nc-4.0",
        "cc-by-nc-sa-4.0",
    }

    if l in permissive:
        return 1.0
    if l in weak or "creative commons" in l or "cc-" in l:
        return 0.6

    # License string exists but is unusual / unknown → minimal credit
    return 0.3


def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    license score ∈ [0,1], HF-only, no Git required.

    1.0 → clearly permissive license (Apache-2.0, MIT, BSD, etc.)
    0.6 → weak copyleft / non-commercial / CC variants
    0.3 → some license string but not recognized as standard permissive
    0.0 → no license information
    """
    start = time.perf_counter()
    score: float = 0.0

    info = _get_model_info(resource)
    if info is not None:
        lic = getattr(info, "license", None)
        card = getattr(info, "cardData", None) or {}
        if not lic:
            lic = card.get("license")
        score = _license_score(lic if isinstance(lic, str) else None)

    score = float(max(0.0, min(1.0, score)))
    latency_ms = int((time.perf_counter() - start) * 1000)
    return score, latency_ms
