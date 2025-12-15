# tests/unit/test_run_extra.py
#
# SWE 45000 – Phase 2
#
# Additional unit tests for src/run.py covering edge cases and
# failure-handling paths not exercised in the primary CLI tests.
#
# These tests validate:
# - Error handling in subprocess execution and dependency installation
# - Robust behavior of process_url_file with missing, empty, and non-model inputs
# - Metric loading when some metrics raise exceptions
# - classify_url edge cases for invalid or non-string inputs
# - Filesystem safety utilities such as remove_readonly
#
# External effects (filesystem, subprocess calls, dynamic imports)
# are mocked or sandboxed to ensure deterministic, isolated execution.
import io
import sys
import run
import types
import builtins
import tempfile
from pathlib import Path
import pytest


def test_run_subprocess_exception(monkeypatch):
    """Covers run_subprocess when subprocess.run raises an exception."""
    def fake_run(cmd, check=False): raise OSError("boom")
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    rc = run.run_subprocess(["echo", "hi"])
    assert rc == 1


def test_handle_install_failure(monkeypatch, tmp_path):
    """Covers handle_install when pip install fails."""
    req = tmp_path / "requirements.txt"
    req.write_text("pkg==0.0.1")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(run, "run_subprocess", lambda *a, **k: 1)
    rc = run.handle_install()
    assert rc == 1


def test_process_url_file_not_found(tmp_path):
    """Covers process_url_file with missing file."""
    missing_file = tmp_path / "nope.csv"
    rc = run.process_url_file(str(missing_file))
    assert rc == 1


def test_process_url_file_empty_and_invalid(tmp_path):
    """Covers process_url_file with empty rows and dataset/code URLs."""
    urlfile = tmp_path / "urls.csv"
    urlfile.write_text(",,\nhttps://github.com/org/repo\nhttps://huggingface.co/datasets/squad\n")

    rc = run.process_url_file(str(urlfile))
    # Should succeed, but no model lines -> no JSON output
    assert rc == 0


def test_compute_metrics_category_and_error(monkeypatch):
    """Covers category metric special case and error in metric call."""
    def fake_category_metric(res): return 0.5, 12
    def fake_bad_metric(res): raise RuntimeError("fail")

    monkeypatch.setitem(sys.modules, "src.metrics.category", types.SimpleNamespace(metric=fake_category_metric))
    monkeypatch.setitem(sys.modules, "src.metrics.bad", types.SimpleNamespace(metric=fake_bad_metric))

    def fake_iter_modules(*a, **k):
        return [(None, "src.metrics.category", False), (None, "src.metrics.bad", False)]

    monkeypatch.setattr(run.pkgutil, "iter_modules", fake_iter_modules)
    metrics = run.load_metrics()
    assert "category" in metrics and "bad" in metrics

    result = run.compute_metrics_for_model({"url": "https://huggingface.co/foo/bar", "name": "foo/bar"})
    assert "category_latency" in result
    assert result["bad"] == 0.0


def test_main_no_args(capsys):
    """Covers main() with no arguments (help text branch)."""
    rc = run.main([])
    out, _ = capsys.readouterr()
    assert rc == 1
    assert "usage:" in out.lower()


@pytest.mark.parametrize("url,expected", [
    (None, "CODE"),       # non-string
    ("", "CODE"),         # empty
    ("https://unknownsite.com/x", "CODE"),
])
def test_classify_url_edges(url, expected):
    assert run.classify_url(url) == expected


def test_remove_readonly(tmp_path):
    """Covers remove_readonly by forcing chmod call."""
    p = tmp_path / "file.txt"
    p.write_text("x")
    # Make read-only
    p.chmod(0o400)
    # Ensure func works even if path is str
    run.remove_readonly(lambda path: Path(path).unlink(), p, None)
    assert not p.exists()
