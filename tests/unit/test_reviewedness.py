# tests/unit/test_reviewedness.py
#
# SWE 45000 – Phase 2
#
# Unit tests for the reviewedness metric.
#
# These tests verify that reviewedness is computed from multiple engagement
# and documentation signals, including:
# - Download count tiers
# - Like count tiers
# - Model card richness (datasets, metrics, language, arXiv references)
#
# The tests ensure:
# - Individual scoring components behave as expected
# - Scores combine correctly and are capped at 1.0
# - Missing or unavailable metadata safely results in a score of 0.0
#
# External API calls are mocked to ensure deterministic, isolated testing.
# tests/unit/test_reviewedness.py
from unittest.mock import patch
from types import SimpleNamespace
import pytest

from src.metrics.reviewedness import metric, _downloads, _likes, _card


class DummyCardData:
    def __init__(self, data):
        self.data = data


def make_info(downloads=0, likes=0, card_dict=None):
    info = SimpleNamespace()
    info.downloads = downloads
    info.likes = likes
    info.cardData = DummyCardData(card_dict) if card_dict is not None else None
    return info


def test_downloads_buckets():
    assert _downloads(make_info(downloads=0)) == 0.0
    assert _downloads(make_info(downloads=500)) == 0.1
    assert _downloads(make_info(downloads=20_000_000)) == 1.0


def test_likes_buckets():
    assert _likes(make_info(likes=0)) == 0.0
    assert _likes(make_info(likes=10)) == 0.4
    assert _likes(make_info(likes=1000)) == 1.0


def test_card_scoring_with_arxiv_and_metrics():
    card = {
        "model-index": [],
        "datasets": [],
        "language": "en",
        "some_ref": "https://arxiv.org/abs/1234.5678",
    }
    info = make_info(card_dict=card)
    score = _card(info)
    # 0.3 base + 0.4(model-index/metrics/evaluation/results) + 0.2(datasets/language/license) + 0.1(arxiv)
    assert pytest.approx(score, rel=1e-6) == 1.0


def test_card_empty_gives_zero():
    info = make_info(card_dict={})
    assert _card(info) == 0.3  # base 0.3 but no extra keys or arxiv


@patch("src.metrics.reviewedness._get_info")
def test_metric_highly_reviewed_model(mock_get_info):
    """
    High downloads, likes, and rich card → score near 1.0.
    """
    card = {
        "model-index": [],
        "datasets": [],
        "language": "en",
        "metrics": [],
        "paper": "https://arxiv.org/abs/1234.5678",
    }
    info = make_info(downloads=25_000_000, likes=2000, card_dict=card)
    mock_get_info.return_value = info

    resource = {"name": "owner/model"}
    score, latency = metric(resource)

    assert 0.9 <= score <= 1.0
    assert isinstance(latency, int)


@patch("src.metrics.reviewedness._get_info")
def test_metric_low_engagement_model(mock_get_info):
    """
    Low downloads, likes, and no card → low reviewedness.
    """
    info = make_info(downloads=0, likes=0, card_dict=None)
    mock_get_info.return_value = info

    resource = {"name": "owner/model"}
    score, _ = metric(resource)

    assert 0.0 <= score <= 0.3


@patch("src.metrics.reviewedness._get_info")
def test_metric_no_info_returns_zero(mock_get_info):
    """
    If _get_info returns None, metric() should be 0.0.
    """
    mock_get_info.return_value = None
    score, latency = metric({"name": "not/hf"})
    assert score == 0.0
    assert isinstance(latency, int)
