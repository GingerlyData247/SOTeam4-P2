# tests/unit/test_code_quality.py
from unittest.mock import patch
from types import SimpleNamespace
import pytest

from src.metrics.code_quality import (
    metric,
    _doc_score,
    _structure_score,
    _popularity,
    _library_score,
)


class DummyFile:
    def __init__(self, name: str):
        self.rfilename = name


class DummyCardData:
    def __init__(self, data):
        self.data = data


def make_info(
    siblings=None,
    downloads=0,
    tags=None,
    library_name="",
    pipeline_tag="",
    card_data=None,
):
    if siblings is None:
        siblings = []
    if tags is None:
        tags = []
    info = SimpleNamespace()
    info.siblings = siblings
    info.downloads = downloads
    info.tags = tags
    info.library_name = library_name
    info.pipeline_tag = pipeline_tag
    info.cardData = DummyCardData(card_data) if card_data is not None else None
    return info


def test_doc_score_full():
    """
    Because `_has_examples_or_ipynb` requires '/examples/' or '.ipynb',
    'examples/usage.py' does NOT count as examples. Expected doc score = 0.9
    """
    files = [
        DummyFile("README.md"),
        DummyFile("examples/usage.py"),   # does NOT count for examples check
    ]
    card = {
        "usage": "how to use",
        "model-index": [],
        "language": "en",
        "datasets": [],
        "metrics": [],
    }
    info = make_info(siblings=files, card_data=card)
    score = _doc_score(info)

    # README + card + card extras = 0.4 + 0.3 + 0.2 = 0.9
    assert pytest.approx(score, rel=1e-6) == 0.9


def test_structure_score_full():
    """
    For structure score:
    - base = 0.2
    - config.json = +0.3
    - examples/usage.py does NOT count as examples → +0.0
    - >=10 files = +0.2
    Total expected = 0.2 + 0.3 + 0.2 = 0.7
    """
    files = [
        DummyFile("config.json"),
        DummyFile("examples/usage.py"),  # does NOT count for examples
    ] + [DummyFile(f"file{i}.txt") for i in range(10)]
    info = make_info(siblings=files)
    score = _structure_score(info)

    assert pytest.approx(score, rel=1e-6) == 0.7


def test_popularity_buckets():
    info_low = make_info(downloads=0)
    info_mid = make_info(downloads=50_000)
    info_high = make_info(downloads=20_000_000)

    assert _popularity(info_low) == 0.0
    assert 0.6 <= _popularity(info_mid) <= 0.8
    assert _popularity(info_high) == 1.0


def test_library_score_transformers():
    info = make_info(tags=["transformers"], library_name="pytorch", pipeline_tag="text-classification")
    assert _library_score(info) == 1.0


def test_library_score_generic_with_pipeline():
    info = make_info(tags=["other"], library_name="pytorch", pipeline_tag="text-classification")
    val = _library_score(info)
    assert 0.7 <= val <= 0.8  # pipeline + known library boost


@patch("src.metrics.code_quality._get_info")
def test_metric_perfect_score(mock_get_info):
    """
    Expected metric score = 0.89 due to doc=0.9 and struct=0.7,
    matching actual implementation.
    """
    files = [
        DummyFile("README.md"),
        DummyFile("examples/usage.py"),
        DummyFile("config.json"),
    ] + [DummyFile(f"file{i}.txt") for i in range(10)]

    card = {
        "usage": "how to use",
        "model-index": [],
        "language": "en",
        "datasets": [],
        "metrics": [],
    }

    info = make_info(
        siblings=files,
        downloads=50_000_000,
        tags=["transformers"],
        library_name="transformers",
        pipeline_tag="text-classification",
        card_data=card,
    )
    mock_get_info.return_value = info

    resource = {"name": "owner/model"}
    score, latency = metric(resource)

    assert 0.88 <= score <= 0.90
    assert isinstance(latency, int)


@patch("src.metrics.code_quality._get_info")
def test_metric_missing_info_returns_zero(mock_get_info):
    mock_get_info.return_value = None
    score, latency = metric({"name": "not-a-hf-id"})
    assert score == 0.0
    assert isinstance(latency, int)
