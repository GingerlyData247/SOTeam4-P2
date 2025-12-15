# tests/unit/test_reproducibility.py
#
# SWE 45000 – Phase 2
#
# Unit tests for the reproducibility metric.
#
# These tests validate that the reproducibility score is computed based on
# the presence of common artifacts that enable experiment replication,
# including dependency files, notebooks, and documentation cues.
#
# The tests ensure:
# - Empty projects yield a score of 0.0
# - Individual reproducibility signals contribute partial credit
# - Multiple signals combine correctly and cap at a maximum score of 1.0
#
# All tests operate on temporary directories to avoid filesystem side effects
# and ensure deterministic behavior.
import pytest
import os
from pathlib import Path
from src.metrics.reproducibility import metric

def test_empty_directory_returns_zero(tmp_path):
    """An empty directory should yield a reproducibility score of 0."""
    score, latency = metric({"local_dir": str(tmp_path)})
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == 0.0


def test_requirements_and_env_files_increase_score(tmp_path):
    """requirements.txt and environment.yml should increase reproducibility."""
    (tmp_path / "requirements.txt").write_text("torch\nnumpy\n", encoding="utf-8")
    (tmp_path / "environment.yml").write_text("dependencies:\n  - pandas\n", encoding="utf-8")

    score, _ = metric({"local_dir": str(tmp_path)})
    # Expect at least 0.6 total (0.4 + 0.2)
    assert score >= 0.6


def test_notebook_adds_reproducibility_points(tmp_path):
    """Presence of .ipynb adds to reproducibility."""
    (tmp_path / "demo.ipynb").write_text("{}", encoding="utf-8")

    score, _ = metric({"local_dir": str(tmp_path)})
    assert 0.1 <= score <= 1.0


def test_readme_with_reproduce_keyword_adds_points(tmp_path):
    """README mentioning 'reproduce' increases score."""
    (tmp_path / "README.md").write_text("How to reproduce our experiments:\n", encoding="utf-8")

    score, _ = metric({"local_dir": str(tmp_path)})
    assert score >= 0.1


def test_combined_features_capped_at_one(tmp_path):
    """Multiple signals combined should cap at 1.0."""
    (tmp_path / "requirements.txt").write_text("torch\n", encoding="utf-8")
    (tmp_path / "environment.yml").write_text("numpy\n", encoding="utf-8")
    (tmp_path / "demo.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("Reproduce the results easily", encoding="utf-8")

    score, _ = metric({"local_dir": str(tmp_path)})
    assert score == pytest.approx(1.0, rel=1e-3)
