# tests/unit/test_license_heuristic.py
from unittest.mock import patch
from src.metrics.license import metric, _license_score


#
# Corrected DummyInfo: cardData must be a DIRECT dict for compatibility
#
class DummyInfo:
    def __init__(self, license_str=None, card_dict=None):
        self.license = license_str
        self.cardData = card_dict  # metric() does `card = cardData or {}`


def test_license_score_permissive():
    for lic in ["mit", "MIT", "apache-2.0", "BSD-3-Clause", "mpl-2.0"]:
        assert _license_score(lic) == 1.0


def test_license_score_weak_or_cc():
    for lic in ["lgpl-3.0", "epl-2.0", "cc-by-nc-4.0", "Creative Commons BY-NC"]:
        assert _license_score(lic) == 0.6


def test_license_score_unknown_nonempty():
    assert _license_score("Some Custom License") == 0.3


def test_license_score_no_license():
    assert _license_score(None) == 0.0
    assert _license_score("") == 0.0


@patch("src.metrics.license._get_model_info")
def test_metric_uses_model_license(mock_info):
    mock_info.return_value = DummyInfo(license_str="mit")
    score, _ = metric({"name": "owner/model"})
    assert 0.9 <= score <= 1.0


@patch("src.metrics.license._get_model_info")
def test_metric_falls_back_to_card_license(mock_info):
    """
    FIXED: cardData must be a dict, since metric() expects cardData.get(...)
    """
    mock_info.return_value = DummyInfo(
        license_str=None,
        card_dict={"license": "apache-2.0"},
    )
    score, _ = metric({"name": "owner/model"})
    assert 0.9 <= score <= 1.0


@patch("src.metrics.license._get_model_info")
def test_metric_no_info_returns_zero(mock_info):
    mock_info.return_value = None
    score, _ = metric({"name": "owner/model"})
    assert score == 0.0
