# tests/unit/test_bus_factor.py
import pytest
from unittest.mock import patch, MagicMock

from src.metrics.bus_factor import compute_bus_factor_from_commits, metric


def test_compute_bus_factor_single_contributor_zero():
    """
    A commit history with only one contributor should yield 0.0
    from compute_bus_factor_from_commits.
    """
    commits = ["alice"] * 20
    score = compute_bus_factor_from_commits(commits)
    assert score == 0.0


def test_compute_bus_factor_multiple_equal_contributors_near_one():
    """
    Evenly distributed contributions among several authors should give
    a bus factor close to 1.0.
    """
    commits = (["alice"] * 10 +
               ["bob"] * 10 +
               ["carol"] * 10 +
               ["dave"] * 10)
    score = compute_bus_factor_from_commits(commits)
    assert pytest.approx(score, rel=1e-3) == 1.0


@patch("src.metrics.bus_factor.requests.get")
def test_metric_uses_github_contributors_count(mock_get):
    """
    metric() should call the GitHub contributors API and map
    the number of contributors to min(1.0, n/10).
    """
    # Fake GitHub API response with 5 contributors
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = [
        {"login": "alice"},
        {"login": "bob"},
        {"login": "carol"},
        {"login": "dave"},
        {"login": "eve"},
    ]
    mock_get.return_value = fake_resp

    resource = {
        "github_url": "https://github.com/owner/repo",
        "url": "https://github.com/owner/repo",
        "name": "owner/repo",
    }

    score, latency = metric(resource)
    # 5 contributors → 0.5
    assert pytest.approx(score, rel=1e-6) == 0.5
    assert isinstance(latency, int) and latency >= 0


@patch("src.metrics.bus_factor.requests.get")
def test_metric_handles_github_api_error_returns_zero(mock_get):
    """
    If the GitHub API fails, the metric should safely return 0.0.
    """
    mock_get.side_effect = Exception("network error")

    resource = {
        "github_url": "https://github.com/owner/repo",
        "url": "https://github.com/owner/repo",
        "name": "owner/repo",
    }

    score, latency = metric(resource)
    assert score == 0.0
    assert isinstance(latency, int) and latency >= 0


def test_metric_no_github_url_returns_zero():
    """
    If there is no GitHub URL and no way to infer one, score should be 0.0.
    """
    resource = {
        "url": "https://huggingface.co/some/model",
        "name": "some/model",
    }
    score, latency = metric(resource)
    assert score == 0.0
    assert isinstance(latency, int)
