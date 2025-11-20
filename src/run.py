#!/usr/bin/env python3

# SWE 45000 – Fall 2025
# TEAM 4 — Phase 1 / Phase 2 shared logic
# This file lives in /src so AWS Lambda can import it.
# The original root-level `run` CLI remains unchanged.

from __future__ import annotations
from src.utils.logging import logger
import argparse
import importlib
import json
import logging
import os
import pkgutil
import subprocess
import sys
import time
import shutil
import stat
import io
import threading
from contextlib import redirect_stdout, redirect_stderr

import tempfile
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable
from urllib.parse import urlparse

from huggingface_hub import snapshot_download
from src.utils.hf_normalize import normalize_hf_id


# ============================================================================
# FIX ROOT PATHS FOR BEING INSIDE src/
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

_prev = os.environ.get("PYTHONPATH", "")
parts = [str(REPO_ROOT), str(SRC_DIR)]
if _prev:
    parts.append(_prev)
os.environ["PYTHONPATH"] = os.pathsep.join(parts)

_METRIC_IMPORT_LOCK = threading.Lock()

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


# ============================================================================
# UTILS
# ============================================================================
def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def run_with_timeout(func, arg, timeout=45, label=None):
    shown = label or getattr(func, "__name__", "unknown")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, arg)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"{shown} timed out after {timeout}s.")
            return 0.0, 0


def _normalize_github_repo_url(url: str) -> str | None:
    try:
        if not url or "github.com" not in url:
            return None
        u = urlparse(url)
        parts = [p for p in u.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            repo = re.sub(r"\.git$", "", repo)
            return f"https://github.com/{owner}/{repo}.git"
    except Exception:
        pass
    return None


# ============================================================================
# INSTALL / TEST SUPPORT (needed to satisfy Phase-1 tests)
# ============================================================================
def run_subprocess(cmd: List[str]) -> int:
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as exc:
        logger.error("Subprocess failed: %s", exc)
        return 1


def _have(mod: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _pip_install(*args: str) -> int:
    return run_subprocess([sys.executable, "-m", "pip", *args])


def ensure_test_deps() -> None:
    if Path("requirements.txt").exists():
        _pip_install("install", "--no-cache-dir", "-r", "requirements.txt")
    if Path("requirements-dev.txt").exists():
        _pip_install("install", "--no-cache-dir", "-r", "requirements-dev.txt")
    if not (_have("pytest") and _have("coverage")):
        _pip_install("install", "--no-cache-dir", "pytest", "pytest-cov", "pytest-mock", "coverage")


def handle_install() -> int:
    rc = 0
    if Path("requirements.txt").exists():
        rc = run_subprocess([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        if rc != 0:
            return rc
    if Path("requirements-dev.txt").exists():
        run_subprocess([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    return rc


def handle_test() -> int:
    import tempfile, xml.etree.ElementTree as ET, importlib.util, subprocess, sys, os

    os.environ.setdefault("PYTHONPATH",
        os.pathsep.join([os.getcwd(), str(Path.cwd() / "src")]))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tf:
        junit_path = tf.name

    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}"],
        text=True, capture_output=True
    )

    passed = total = 0
    try:
        root = ET.parse(junit_path).getroot()
        node = root if root.tag == "testsuite" else root.find("testsuite")
        if node is not None:
            tests   = int(node.attrib.get("tests", "0"))
            fails   = int(node.attrib.get("failures", "0"))
            errors  = int(node.attrib.get("errors", "0"))
            skipped = int(node.attrib.get("skipped", "0"))
            total   = tests
            passed  = tests - fails - errors - skipped
    except Exception:
        out = p.stdout or ""
        passed = out.count(".")
        total  = max(passed, 0)

    cov_pct = "0"
    if importlib.util.find_spec("coverage") is not None:
        subprocess.run([sys.executable, "-m", "coverage", "erase"],
                       text=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "coverage", "run", "-m", "pytest", "-q"],
                       text=True, capture_output=True)
        rep = subprocess.run([sys.executable, "-m", "coverage", "report", "-m"],
                             text=True, capture_output=True)
        for ln in (rep.stdout or "").splitlines()[::-1]:
            parts = ln.split()
            if parts and parts[-1].endswith("%"):
                cov_pct = parts[-1].rstrip("%")
                break

    print(f"{passed}/{total} test cases passed. {cov_pct}% line coverage achieved.")
    return 0


# ============================================================================
# URL Classification
# ============================================================================
def classify_url(url: str) -> str:
    if not isinstance(url, str):
        return "CODE"
    u = url.strip()
    if not u:
        return "CODE"

    p = urlparse(u)
    host = (p.netloc or "").lower()
    path = (p.path or "").lower().lstrip("/")

    if host.endswith("huggingface.co"):
        if path.startswith("datasets/"):
            return "DATASET"
        return "MODEL"

    if host in {"github.com", "gitlab.com", "bitbucket.org"}:
        return "CODE"

    return "CODE"


# ============================================================================
# Metric Loader
# ============================================================================
def load_metrics() -> Dict[str, Callable[[Dict[str, Any]], Tuple[float, int]]]:
    metrics: Dict[str, Callable] = {}
    metrics_pkg = "src.metrics"
    try:
        package = importlib.import_module(metrics_pkg)
    except ModuleNotFoundError:
        return metrics

    with _METRIC_IMPORT_LOCK:
        for _, mod_name, is_pkg in pkgutil.iter_modules(package.__path__,
                package.__name__ + "."):
            if is_pkg:
                continue
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    module = importlib.import_module(mod_name)
            except Exception:
                continue
            func = getattr(module, "metric", None)
            if callable(func):
                metrics[mod_name.split(".")[-1]] = func
    return metrics


# ============================================================================
# Compute Metrics
# ============================================================================
def compute_metrics_for_model(resource: Dict[str, Any]) -> Dict[str, Any]:
    metrics = load_metrics()

    out = {
        "name": resource.get("name", "unknown"),
        "category": "MODEL",
    }

    results: Dict[str, Tuple[float, int]] = {}

    for name, func in metrics.items():
        if resource.get("skip_repo_metrics") and name in {"bus_factor", "code_quality"}:
            continue

        try:
            score, latency = run_with_timeout(func, resource, timeout=90,
                                              label=f"metric:{name}")
            if isinstance(score, (int, float)):
                score = float(max(0.0, min(1.0, score)))
        except Exception:
            score, latency = 0.0, 0

        results[name] = (score, latency)

    for name, (score, latency) in results.items():
        if name == "size_score":
            out[name] = {
                "raspberry_pi": score,
                "jetson_nano": score,
                "desktop_pc": score,
                "aws_server": score,
            }
        else:
            out[name] = score
        out[f"{name}_latency"] = latency

    numeric = [s for (s, _) in results.values() if isinstance(s, (int, float))]
    net_latency = sum(lat for (_, lat) in results.values())
    out["net_score"] = round(sum(numeric) / len(numeric), 4) if numeric else 0.0
    out["net_score_latency"] = net_latency

    return out


# ============================================================================
# SAFE REPO SETUP (Phase 2 → NO real cloning)
# ============================================================================
def _safe_repo_setup(r, find_github_url_from_hf, clone_repo_to_temp):
    r["skip_repo_metrics"] = False
    r["local_path"] = None
    return r


# ============================================================================
# URL File Processing (CLI)
# ============================================================================
def process_url_file(path_str: str) -> int:
    try:
        from src.utils.repo_cloner import clone_repo_to_temp
    except ImportError:
        clone_repo_to_temp = None
    try:
        from src.utils.github_link_finder import find_github_url_from_hf
    except ImportError:
        find_github_url_from_hf = None

    p = Path(path_str)
    if not p.exists():
        print(f"Error: URL file not found: {path_str}", file=sys.stderr)
        return 1

    # --- FIX #1: Match Phase-1 behavior – split by commas & newlines ---
    raw = p.read_text(encoding="utf-8")
    urls = [t.strip()
            for line in raw.splitlines()
            for t in line.split(",")
            if t.strip()]

    if not urls:
        return 0

    # Build resource dicts
    resources = []
    for u in urls:
        resources.append({
            "url": u,
            "category": classify_url(u),
            "name": (
                "/".join(u.rstrip("/").split("/")[-2:])
                if "github.com" in u
                else normalize_hf_id(u)
            ),
        })

    models = [r for r in resources if r["category"] == "MODEL"]

    # HF metadata
    from huggingface_hub import HfApi
    api = HfApi()

    for r in models:
        repo_id = r["name"]
        try:
            info = api.model_info(repo_id)
        except Exception:
            info = None

        r["is_hf"] = True
        r["hf_license"] = getattr(info, "license", None) if info else None
        r["hf_metadata"] = {
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "tags": getattr(info, "tags", []),
            "downloads": getattr(info, "downloads", 0),
        }

    for r in models:
        _safe_repo_setup(r, find_github_url_from_hf, clone_repo_to_temp)

    # Metric execution
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as exe:
        futures = {exe.submit(compute_metrics_for_model, r): r for r in models}
        for fut in as_completed(futures):
            try:
                result = fut.result()
                sys.stdout.write(json.dumps(result, ensure_ascii=False,
                                            separators=(",", ":")) + "\n")
            except Exception:
                continue

    return 0


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run",
        description="Phase 1/2 unified CLI")
    parser.add_argument("arg", nargs="?", help="install | test | URL_FILE")
    args = parser.parse_args(argv)

    if args.arg is None:
        parser.print_help()
        return 1

    if args.arg == "install":
        return handle_install()
    if args.arg == "test":
        return handle_test()

    return process_url_file(args.arg)


if __name__ == "__main__":
    sys.exit(main())