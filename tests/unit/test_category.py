# ---------------------------------------------------------------------------
# Unit Tests: Category Metric
#
# This test suite verifies the behavior of the `category` metric, which
# classifies artifacts based on their source and metadata.
#
# Covered behavior:
#   - Hugging Face models using `pipeline_tag` as the category when available
#   - Default categorization for Hugging Face models without a pipeline tag
#   - Correct classification of GitHub repositories as code artifacts
#
# External API calls to Hugging Face are mocked to ensure deterministic,
# fast, and network-independent test execution.
#
# These tests support the Phase 2 Trustworthy Model Registry by ensuring
# consistent artifact categorization across model and code sources.
# ---------------------------------------------------------------------------
from src.metrics.category import metric

# A fake class to simulate the model_info object from the real API
class FakeModelInfo:
    def __init__(self, pipeline_tag=None):
        self.pipeline_tag = pipeline_tag

def test_category_for_hf_model_with_tag(mocker):
    """Tests that a Hugging Face model gets its pipeline_tag as the category."""
    # Fake the API call to return an object with a specific pipeline_tag
    mocker.patch('src.metrics.category.model_info', return_value=FakeModelInfo("text-generation"))

    resource = {"url": "https://huggingface.co/gpt2", "name": "gpt2"}
    category, latency = metric(resource)

    assert category == "text-generation"

def test_category_for_hf_model_without_tag(mocker):
    """Tests that a Hugging Face model without a tag gets a default category."""
    # Fake the API call to return an object with no pipeline_tag
    mocker.patch('src.metrics.category.model_info', return_value=FakeModelInfo(None))

    resource = {"url": "https://huggingface.co/some-model", "name": "some-model"}
    category, latency = metric(resource)

    assert category == "Model"

def test_category_for_github_repo(mocker):
    """Tests that a GitHub URL is correctly categorized."""
    # Mock the API call just in case, though it shouldn't be called for a GitHub URL
    mocker.patch('src.metrics.category.model_info')

    resource = {"url": "https://github.com/expressjs/express", "name": "expressjs/express"}
    category, latency = metric(resource)

    assert category == "Code Repository"
