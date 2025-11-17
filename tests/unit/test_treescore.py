# tests/unit/test_treescore.py
import pytest
from unittest.mock import patch

import src.metrics.treescore as treescore


@patch("src.metrics.treescore._parents")
@patch("src.metrics.treescore._net")
def test_no_parents_uses_own_net_score(mock_net, mock_parents):
    """
    If a model has no parents, treescore.metric should return its own net score.
    """
    mock_parents.return_value = []
    mock_net.return_value = 0.8

    res = {"name": "owner/model", "url": "https://huggingface.co/owner/model"}
    score, lat = treescore.metric(res)

    assert pytest.approx(score, rel=1e-3) == 0.8
    assert isinstance(lat, int) and lat >= 0


@patch("src.metrics.treescore._parents")
@patch("src.metrics.treescore._net")
def test_parents_contribute_to_score(mock_net, mock_parents):
    """
    When parents exist, the score should combine the model's own net score
    with the average of parents' net scores.
    """

    def fake_parents(repo: str):
        if repo == "child/model":
            return ["p1", "p2"]
        else:
            return []

    def fake_net(repo: str) -> float:
        if repo == "child/model":
            return 0.2
        if repo == "p1":
            return 0.5
        if repo == "p2":
            return 1.0
        return 0.0

    mock_parents.side_effect = fake_parents
    mock_net.side_effect = fake_net

    res = {"name": "child/model", "url": "https://huggingface.co/child/model"}
    score, _ = treescore.metric(res)

    # parents avg = (0.5 + 1.0) / 2 = 0.75
    # final score = (own 0.2 + parents_avg 0.75) / 2 = 0.475
    assert pytest.approx(score, rel=1e-3) == 0.475


@patch("src.metrics.treescore._parents")
@patch("src.metrics.treescore._net")
def test_cycle_detection_does_not_infinite_recuse(mock_net, mock_parents):
    """
    If parents create a cycle (A -> B -> A), the 'seen' set should prevent
    infinite recursion, and the metric should still return a finite score.
    """

    def fake_parents(repo: str):
        if repo == "child/model":
            return ["parent/model"]
        if repo == "parent/model":
            return ["child/model"]
        return []

    def fake_net(repo: str) -> float:
        if repo == "child/model":
            return 0.3
        if repo == "parent/model":
            return 0.9
        return 0.0

    mock_parents.side_effect = fake_parents
    mock_net.side_effect = fake_net

    res = {"name": "child/model", "url": "https://huggingface.co/child/model"}
    score, lat = treescore.metric(res)

    assert 0.0 <= score <= 1.0
    assert isinstance(lat, int) and lat >= 0
