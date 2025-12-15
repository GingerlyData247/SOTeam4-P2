# ---------------------------------------------------------------------------
# Unit Tests: Dataset Quality Metric
#
# This test suite verifies the correctness and robustness of the dataset
# quality metric, which evaluates the reliability and usefulness of datasets
# associated with a Hugging Face model.
#
# The tests cover:
#   - Extraction and normalization of dataset identifiers from URLs
#   - Scoring of individual datasets using Hugging Face metadata
#     (e.g., downloads, likes, and dataset cards)
#   - Combination of dataset signals derived from model tags and README parsing
#   - Graceful handling of invalid dataset references and API failures
#
# All external Hugging Face API interactions are mocked to ensure deterministic,
# offline-safe execution and to comply with the Phase 2 Trustworthy Model Registry
# grading and reproducibility requirements.
# ---------------------------------------------------------------------------
import pytest
from unittest.mock import patch, MagicMock
from src.metrics import dataset_quality as dq

def test_get_dataset_id_from_url():
    url = "https://huggingface.co/datasets/rajpurkar/squad"
    assert dq.get_dataset_id_from_url(url) == "rajpurkar/squad"
    assert dq.get_dataset_id_from_url("https://example.com") is None

@patch("src.metrics.dataset_quality.dataset_info")
def test_score_single_dataset_valid(mock_dataset_info):
    mock_info = MagicMock()
    mock_info.cardData = {"some": "data"}
    mock_info.downloads = 5000
    mock_info.likes = 50
    mock_dataset_info.return_value = mock_info

    score = dq.score_single_dataset("rajpurkar/squad", token=None)
    assert 0.5 <= score <= 1.0  # Should combine partial weights

@patch("src.metrics.dataset_quality.dataset_info", side_effect=Exception("repo not found"))
def test_score_single_dataset_invalid(mock_dataset_info):
    score = dq.score_single_dataset("invalid/data", token=None)
    assert score == 0.0

@patch("src.metrics.dataset_quality.find_datasets_from_resource")
@patch("src.metrics.dataset_quality.HfApi")
def test_metric_combines_tag_and_readme(mock_api, mock_find):
    # Mock model_info to include tags
    mock_api.return_value.model_info.return_value.tags = ["dataset:rajpurkar/squad"]
    # Mock find_datasets_from_resource to include an additional dataset
    mock_find.return_value = (["rajpurkar/squad"], 123)

    resource = {"name": "google-bert/bert-base-uncased"}
    score, latency = dq.metric(resource)

    assert isinstance(score, float)
    assert isinstance(latency, int)

@patch("src.metrics.dataset_quality.find_datasets_from_resource", return_value=([], 0))
@patch("src.metrics.dataset_quality.HfApi")
def test_metric_handles_api_failure(mock_api, mock_find):
    instance = mock_api.return_value
    instance.model_info.side_effect = Exception("api failure")

    resource = {"name": "fake/model"}
    score, latency = dq.metric(resource)

    assert score == 0.0
    assert isinstance(latency, int)
