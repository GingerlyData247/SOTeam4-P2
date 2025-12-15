# ---------------------------------------------------------------------------
# Unit Tests: Performance Claims Metric
#
# This test suite verifies the correctness and robustness of the
# `performance_claims` metric, which estimates model credibility based on
# Hugging Face download statistics.
#
# The tests validate:
#   - Correct tier-based score assignment across all download thresholds
#   - Proper handling of Hugging Face API errors (HfHubHTTPError)
#   - Graceful failure behavior for generic exceptions and missing metadata
#   - Deterministic latency reporting under all conditions
#
# All external API calls are fully mocked to ensure deterministic,
# offline-safe execution and compliance with the Phase 2 grading rubric.
# ---------------------------------------------------------------------------
# tests/unit/test_performance_claims.py

from src.metrics.performance_claims import metric
from requests import Response
import pytest
from huggingface_hub.utils import HfHubHTTPError

class FakeModelInfo:
    def __init__(self, downloads):
        self.downloads = downloads


def test_performance_claims_high_downloads(mocker):
    """Models with >1M downloads should return 1.0 score."""
    mocker.patch('src.metrics.performance_claims.model_info',
                 return_value=FakeModelInfo(5_000_000))
    score, latency = metric({"name": "google/gemma-2b"})
    assert score == 1.0
    assert latency >= 0


def test_performance_claims_api_error(mocker):
    """If HuggingFace API raises HfHubHTTPError, score must be 0.0."""
    from requests import Response

    # Create a dummy Response object to satisfy the new constructor
    dummy_response = Response()
    dummy_response.status_code = 404

    mocker.patch(
        "src.metrics.performance_claims.model_info",
        side_effect=HfHubHTTPError(message="Not Found", response=dummy_response),
    )
    score, latency = metric({"name": "not/a-real-model"})
    assert score == 0.0
    assert latency >= 0


@pytest.mark.parametrize("downloads,expected", [
    (0, 0.1),       # very low
    (150, 0.2),     # >100
    (1500, 0.4),    # >1000
    (15000, 0.6),   # >10000
    (150000, 0.8),  # >100000
])
def test_performance_claims_download_tiers(mocker, downloads, expected):
    """Covers all branches of the tiered scoring logic."""
    mocker.patch('src.metrics.performance_claims.model_info',
                 return_value=FakeModelInfo(downloads))
    score, latency = metric({"name": "some/model"})
    assert score == expected
    assert latency >= 0


def test_performance_claims_missing_name():
    """If resource lacks 'name', function should return score=0.0."""
    score, latency = metric({})
    assert score == 0.0
    assert latency >= 0


def test_performance_claims_generic_exception(mocker):
    """If model_info raises a generic Exception, catch-all should return 0.0."""
    mocker.patch('src.metrics.performance_claims.model_info',
                 side_effect=RuntimeError("Network down"))
    score, latency = metric({"name": "any/model"})
    assert score == 0.0
    assert latency >= 0
