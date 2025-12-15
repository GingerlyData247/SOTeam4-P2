# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# METRIC: dataset_quality
# REQUIREMENTS SATISFIED: dataset_quality metric score

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/metrics/dataset_quality.py

Computes the Dataset Quality metric for a model artifact.

This metric evaluates the quality and usability of datasets associated
with a model by identifying dataset references from multiple sources,
including Hugging Face model tags and README content. Each discovered
dataset is scored based on available metadata such as documentation,
download count, and community engagement.

Scoring approach:
    - Dataset references are extracted from model tags and README links
    - Each dataset is scored individually using Hugging Face metadata
    - The final score is the maximum dataset score found
    - If datasets are referenced but not scorable, a partial fallback
      score is returned

The metric follows the Phase 2 specification and is designed to be
fault-tolerant, always returning a bounded score and measured latency
even when metadata retrieval fails.
"""
from __future__ import annotations
import time, re, logging, os
from typing import Any, Dict, Tuple, Set
from urllib.parse import urlparse
from dotenv import load_dotenv
from huggingface_hub import HfApi, dataset_info, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HFValidationError
from src.utils.dataset_link_finder import find_datasets_from_resource, _normalize_dataset_ref
from src.utils.hf_normalize import normalize_hf_id

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
logger = logging.getLogger("phase1_cli")


def get_dataset_id_from_url(url: str) -> str | None:
    try:
        parts = urlparse(url).path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "datasets":
            return _normalize_dataset_ref("/".join(parts[1:3]))
    except Exception:
        pass
    return None


def score_single_dataset(ds_id: str, token: str | None) -> float:
    try:
        info = dataset_info(ds_id, token=token)
        card = 0.5 if getattr(info, "cardData", None) else 0.0
        downloads = 0.3 if getattr(info, "downloads", 0) > 1000 else 0.0
        likes = 0.2 if getattr(info, "likes", 0) > 10 else 0.0
        return card + downloads + likes
    except Exception as e:
        logger.debug(f"DatasetQuality: fail for {ds_id}: {e}")
        return 0.0


def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    start = time.perf_counter()
    repo_id = resource.get("name")
    all_refs: Set[str] = set()
    token = HF_TOKEN
    api = HfApi(token=token)

    # --- Step 1: tags from model_info ---
    try:
        info = api.model_info(normalize_hf_id(repo_id))
        tags = getattr(info, "tags", [])
        tag_refs = {t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("dataset:")}
        for ref in tag_refs:
            if norm := _normalize_dataset_ref(ref):
                all_refs.add(norm)
    except Exception:
        pass

    # --- Step 2: README parse ---
    try:
        found, _ = find_datasets_from_resource(resource)
        all_refs.update(found or [])
    except Exception as e:
        logger.debug(f"DatasetQuality: README parse fail for {repo_id}: {e}")

    # --- Step 3: scoring ---
    scorable = {r for r in all_refs if "/" in r}
    scores = [score_single_dataset(r, token) for r in scorable] if scorable else []
    final = max(scores) if scores else (0.5 if all_refs else 0.0)

    latency = int((time.perf_counter() - start) * 1000)
    return round(min(final, 1.0), 3), latency
