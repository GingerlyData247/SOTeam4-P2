# tests/unit/test_run_cli.py
#
# SWE 45000 – Phase 2
#
# Unit tests for the unified CLI logic in src/run.py.
#
# These tests cover:
# - URL classification logic for models, datasets, and code repositories
# - CLI command branches (install, test, URL file processing)
# - Graceful handling of missing input files and invalid arguments
# - Dynamic metric loading and aggregation via compute_metrics_for_model
#
# External dependencies (filesystem access, subprocess calls, metrics,
# and network-related utilities) are mocked to ensure deterministic,
# fast, and isolated test execution.
import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import run


def test_classify_url_variants():
    """Covers classify_url for HF model, dataset, code, and invalid inputs."""
    assert run.classify_url("https://huggingface.co/datasets/squad") == "DATASET"
    assert run.classify_url("https://huggingface.co/google/bert-base") == "MODEL"
    assert run.classify_url("https://github.com/user/repo") == "CODE"
    assert run.classify_url("not-a-url") == "CODE"
    assert run.classify_url("") == "CODE"
    assert run.classify_url(None) == "CODE"


def test_handle_install_and_test(monkeypatch, tmp_path):
    """Covers handle_install and handle_test branches."""
    # fake requirements.txt
    req = tmp_path / "requirements.txt"
    req.write_text("pytest\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "run_subprocess", lambda *a, **k: 0)

    assert run.handle_install() == 0

    # patch subprocess for handle_test
    monkeypatch.setattr(run, "run_subprocess", lambda *a, **k: 0)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="..", returncode=0))
    monkeypatch.setattr(run.importlib.util, "find_spec", lambda name: True)

    assert run.handle_test() == 0


def test_process_url_file_missing(tmp_path, capsys):
    """Missing file should print error and return 1."""
    missing_file = tmp_path / "nofile.csv"
    rc = run.process_url_file(str(missing_file))
    captured = capsys.readouterr()
    assert "Error: URL file not found" in captured.err
    assert rc == 1


def test_process_url_file_with_model(monkeypatch, tmp_path):
    """Covers process_url_file with a fake HF model URL."""
    urlfile = tmp_path / "urls.csv"
    urlfile.write_text(",,https://huggingface.co/google/bert-base-uncased\n")

    # patch compute_metrics_for_model
    monkeypatch.setattr(run, "compute_metrics_for_model", lambda r: {"name": r["name"], "url": r["url"], "net_score": 1.0, "net_score_latency": 5})

    # patch clone_repo_to_temp in its real module
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    sys.modules["src.utils.repo_cloner"] = types.SimpleNamespace(clone_repo_to_temp=lambda url: str(fake_repo))

    # patch find_github_url_from_hf
    sys.modules["src.utils.github_link_finder"] = types.SimpleNamespace(find_github_url_from_hf=lambda name: "https://github.com/google/research")

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    rc = run.process_url_file(str(urlfile))
    assert rc == 0
    result = json.loads(out.getvalue().strip())
    assert result["net_score"] == 1.0


def test_load_metrics_and_compute(monkeypatch):
    """Covers load_metrics and compute_metrics_for_model with fake modules."""
    # fake huggingface_service module to satisfy model_scorer import
    sys.modules["huggingface_service"] = types.SimpleNamespace(ModelMetadata=object)

    # create a fake metric module
    fake_module = types.SimpleNamespace(metric=lambda r: (0.7, 10))
    sys.modules["src.metrics.fake_metric"] = fake_module

    with patch("pkgutil.iter_modules", return_value=[(None, "src.metrics.fake_metric", False)]):
        metrics = run.load_metrics()
        assert "fake_metric" in metrics

    resource = {"url": "https://huggingface.co/google/bert-base-uncased", "name": "google/bert-base-uncased"}
    result = run.compute_metrics_for_model(resource)
    assert "net_score" in result
    assert isinstance(result["net_score_latency"], int)


def test_main_branches(monkeypatch, tmp_path, capsys):
    """Covers main() with install, test, and missing args."""
    monkeypatch.setattr(run, "handle_install", lambda: 0)
    monkeypatch.setattr(run, "handle_test", lambda: 0)
    monkeypatch.setattr(run, "process_url_file", lambda path: 0)

    assert run.main(["install"]) == 0
    assert run.main(["test"]) == 0

    urlfile = tmp_path / "urls.csv"
    urlfile.write_text("https://github.com/user/repo\n")
    assert run.main([str(urlfile)]) == 0

    # missing arg prints help
    rc = run.main([])
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert rc == 1
