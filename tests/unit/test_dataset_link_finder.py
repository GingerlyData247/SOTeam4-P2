# ---------------------------------------------------------------------------
# Unit Tests: Dataset Link Finder Utility
#
# This test suite validates the behavior of the dataset link discovery utilities
# used to identify training datasets associated with Hugging Face model artifacts.
#
# The tests cover:
#   - Normalization of dataset references from raw strings and URLs
#   - Extraction of dataset identifiers from Hugging Face model metadata
#   - Fallback discovery of dataset links by parsing README content
#   - Safe degradation and empty results when metadata or README data is missing
#
# External Hugging Face API calls and README downloads are mocked to ensure
# deterministic execution and to verify robustness against network or
# repository failures, as required by the Phase 2 Trustworthy Model Registry
# specification.
# ---------------------------------------------------------------------------
import pytest
from unittest.mock import patch, MagicMock
from src.utils import dataset_link_finder as dlf

def test_normalize_dataset_ref_variants():
    # Valid direct IDs
    assert dlf._normalize_dataset_ref("squad") == "squad"
    assert dlf._normalize_dataset_ref("rajpurkar/squad") == "rajpurkar/squad"
    # Valid URLs
    assert dlf._normalize_dataset_ref("https://huggingface.co/datasets/rajpurkar/squad") == "rajpurkar/squad"
    # Invalid formats
    assert dlf._normalize_dataset_ref("https://example.com") is None
    assert dlf._normalize_dataset_ref("") is None
    assert dlf._normalize_dataset_ref(None) is None

@patch("src.utils.dataset_link_finder.model_info")
def test_find_datasets_from_metadata(mock_model_info):
    mock_info = MagicMock()
    mock_info.datasets = ["rajpurkar/squad", "invalid-dataset"]
    mock_model_info.return_value = mock_info

    resource = {"name": "google-bert/bert-base-uncased", "local_dir": None}
    found, latency = dlf.find_datasets_from_resource(resource)
    assert "rajpurkar/squad" in found
    assert isinstance(latency, int)
    assert latency >= 0

@patch("src.utils.dataset_link_finder._fetch_readme_from_hf")
@patch("src.utils.dataset_link_finder.model_info", side_effect=Exception("ignore"))
def test_find_datasets_from_readme(mock_model_info, mock_fetch_readme):
    # README includes an HF dataset URL
    mock_fetch_readme.return_value = "Trained on [SQuAD](https://huggingface.co/datasets/rajpurkar/squad)"
    resource = {"name": "test/model", "local_dir": None}

    found, latency = dlf.find_datasets_from_resource(resource)
    assert "rajpurkar/squad" in found
    assert isinstance(latency, int)

@patch("src.utils.dataset_link_finder._fetch_readme_from_hf", return_value=None)
@patch("src.utils.dataset_link_finder.model_info", side_effect=Exception("not found"))
def test_find_datasets_returns_empty_for_invalid_repo(mock_model_info, mock_fetch):
    resource = {"name": "fake/model", "local_dir": None}
    found, latency = dlf.find_datasets_from_resource(resource)
    assert found == []
    assert isinstance(latency, int)
