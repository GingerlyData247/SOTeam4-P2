# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# METRIC: reproducibility
# REQUIREMENTS SATISFIED: reproducibility metric score

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/metrics/reproducibility.py

Computes the Reproducibility metric for a model artifact.

This metric evaluates how easily a model’s results can be reproduced by
examining the availability of environment configuration files, example
notebooks, and reproducibility instructions. When possible, the metric
prefers inspecting a locally available repository. If local inspection
is not possible, it falls back to lightweight metadata-based heuristics
using remote Hugging Face information.

Scoring approach:
    - Local repository analysis (preferred):
        • requirements files
        • environment configuration files
        • Jupyter notebooks
        • README instructions mentioning reproduction
    - Remote fallback:
        • presence of training, dataset, or configuration metadata

The metric is fault-tolerant and deterministic, always returning a
bounded score and measured latency per the Phase 2 specification.
"""
# src/metrics/reproducibility.py
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import requests

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _score_local_reproducibility(local_dir: str) -> float:
    """
    Inspect local repository files to infer reproducibility signals.
    Weights:
      requirements.txt   → +0.4
      environment.yml    → +0.2
      .ipynb notebook    → +0.2
      README with 'reproduce' → +0.2
    """
    score = 0.0
    try:
        p = Path(local_dir)
        if not p.exists():
            return 0.0

        # requirements.txt
        if any(f.name.lower().startswith("requirements") for f in p.iterdir()):
            score += 0.4

        # environment.yml
        if any(f.name.lower().startswith("environment") and f.suffix in (".yml", ".yaml") for f in p.iterdir()):
            score += 0.2

        # Jupyter notebooks
        if any(f.suffix.lower() == ".ipynb" for f in p.iterdir()):
            score += 0.2

        # README mentions "reproduce"
        for readme in p.glob("README*"):
            try:
                text = readme.read_text(encoding="utf-8", errors="ignore").lower()
                if "reproduce" in text:
                    score += 0.2
                    break
            except Exception:
                continue
    except Exception:
        return 0.0

    return min(score, 1.0)


def _score_remote_reproducibility(resource: Dict[str, Any]) -> float:
    """
    Remote fallback using metadata or repo inspection.
    """
    url = resource.get("url", "")
    if "huggingface.co" in url:
        try:
            repo_id = url.split("huggingface.co/")[-1].strip("/")
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            r = requests.get(f"https://huggingface.co/api/models/{repo_id}", headers=headers, timeout=10)
            if r.status_code == 200:
                info = r.json()
                # Heuristic: if model has 'training', 'datasets', or 'config' info, it’s reproducible
                if any(k in info for k in ("training", "datasets", "config")):
                    return 0.8
        except Exception:
            return 0.0
    return 0.0


def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Combined reproducibility metric:
      • Prefer local_dir analysis (for tests and cloned repos)
      • Otherwise fall back to Hugging Face / GitHub heuristics
    """
    start = time.perf_counter()
    local_dir = resource.get("local_dir") or resource.get("local_path")

    # ✅ 1. Local analysis always takes precedence
    if local_dir and os.path.isdir(local_dir):
        score = _score_local_reproducibility(local_dir)
        latency = int((time.perf_counter() - start) * 1000)
        return round(score, 3), latency

    # ✅ 2. Remote fallback (only if no local_dir)
    score = _score_remote_reproducibility(resource)
    latency = int((time.perf_counter() - start) * 1000)
    return round(score, 3), latency
