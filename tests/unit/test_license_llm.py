# tests/unit/test_license_llm.py
from unittest.mock import patch
from src.metrics.license import metric, _license_score


class DummyCardData:
    def __init__(self, data):
        self.data = data


class DummyInfo:
    def __init__(self, license_str=None, card_data=None):
        self.license = license_str
        self.cardData = DummyCardData(card_data) if card_data is not None else None


@patch("src.metrics.license._get_model_info")
def test_metric_case_insensitive(mock_get_info):
    """
    License matching must be case-insensitive.
    """
    mock_get_info.return_value = DummyInfo(license_str="Apache-2.0")
    resource = {"name": "owner/model"}

    score, _ = metric(resource)
    # Apache should still be treated as permissive (score near 1.0)
    assert 0.8 <= score <= 1.0


@patch("src.metrics.license._get_model_info")
def test_metric_creative_commons_treated_as_weak(mock_get_info):
    """
    Creative Commons licenses should be treated as weakly compatible (0.6).
    """
    mock_get_info.return_value = DummyInfo(license_str="CC-BY-NC-SA-4.0")
    resource = {"name": "owner/model"}

    score, _ = metric(resource)
    assert 0.5 <= score <= 0.7  # around 0.6


def test_license_score_direct_creative_commons():
    """
    Direct calls to _license_score for CC licenses should yield ~0.6.
    """
    val = _license_score("Creative Commons BY-NC-SA 4.0")
    assert 0.5 <= val <= 0.7

