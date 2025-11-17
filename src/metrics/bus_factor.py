# src/metrics/bus_factor.py
from __future__ import annotations
import time, re, math, os
from typing import Dict, Tuple, List
from collections import Counter
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
from src.utils.logging import logger
from src.utils.github_link_finder import find_github_url_from_hf

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _normalize_github_repo_url(url: str | None) -> str | None:
    if not url or "github.com" not in url:
        return None
    try:
        u = urlparse(url)
        parts = [p for p in u.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1].replace(".git", "")
            return f"https://github.com/{owner}/{repo}"
    except Exception:
        pass
    return None


def compute_bus_factor_from_commits(commits: List[str]) -> float:
    if not commits:
        return 0.0
    counts = Counter(commits)
    n = len(counts)
    if n <= 1:
        return 0.0
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy / math.log2(n)


def metric(resource: Dict) -> Tuple[float, int]:
    start = time.perf_counter()

    raw = resource.get("github_url")
    if not raw and "huggingface.co" in resource.get("url", ""):
        try:
            gh = find_github_url_from_hf(resource.get("name", ""))
            if gh:
                raw = gh
        except Exception:
            raw = None

    if not raw or "github.com" not in raw:
        latency = int((time.perf_counter() - start) * 1000)
        return 0.0, latency

    repo_url = _normalize_github_repo_url(raw)
    if not repo_url:
        latency = int((time.perf_counter() - start) * 1000)
        return 0.0, latency

    try:
        parts = urlparse(repo_url).path.strip("/").split("/")
        repo_id = "/".join(parts[:2])
    except Exception:
        latency = int((time.perf_counter() - start) * 1000)
        return 0.0, latency

    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo_id}/contributors",
            headers=headers,
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                score = min(1.0, len(data) / 10)
                latency = int((time.perf_counter() - start) * 1000)
                return round(score, 3), latency
        else:
            logger.debug(f"BusFactor: API status={r.status_code}")
    except Exception as e:
        logger.debug(f"BusFactor: GitHub API error: {e}")

    latency = int((time.perf_counter() - start) * 1000)
    return 0.0, latency
