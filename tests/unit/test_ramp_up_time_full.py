# tests/unit/test_ramp_up_time_full.py
#
# Unit tests for the ramp_up_time metric.
#
# This file validates:
# - Boundary behavior of the internal _length_score helper
# - Local README discovery across multiple filename variants
# - Detection of installation instructions and code examples
# - Score clamping at the maximum value of 1.0
# - Graceful handling of missing files and unexpected errors
# - Remote README fetching logic for GitHub repositories,
#   including main → master fallback and network failures
#
# All tests are deterministic and use temporary directories
# or mocking to avoid real filesystem or network dependencies.
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Make sure to import the metric function from your src directory
from src.metrics.ramp_up_time import metric, _length_score

# Helper function to create temporary README files
def write_readme(tmp_path: Path, text: str, name: str = "README.md") -> str:
    """Creates a README file in a temporary directory and returns the path."""
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(tmp_path)

# --- 1. Tests for the _length_score function ---

@pytest.mark.parametrize("words, expected_score", [
    (49, 0.0),      # Below threshold
    (50, 0.1),      # Lower boundary
    (199, 0.1),     # Upper boundary
    (200, 0.25),    # Lower boundary
    (499, 0.25),    # Upper boundary
    (500, 0.4),     # Lower boundary
])
def test_length_score_boundaries(words, expected_score):
    """Ensures all branches of the _length_score function are covered."""
    assert _length_score(words) == expected_score

# --- 2. Tests for Local File Logic ---

def test_no_readme_present(tmp_path):
    """Covers the case where no README file is found in the local directory."""
    score, _ = metric({"local_dir": str(tmp_path)})
    assert score == 0.0

def test_read_different_readme_names(tmp_path):
    """Covers the logic for finding READMEs with different filenames (e.g., .rst)."""
    text = "word " * 55  # Score 0.1 for length
    d = write_readme(tmp_path, text, name="README.rst")
    score, _ = metric({"local_dir": d})
    assert score == 0.1

def test_full_score_and_clamping(tmp_path):
    """Covers a scenario where all scoring components are met, testing the max score of 1.0."""
    text = """
    # My Project
    This project is great. ('word ' * 500)
    ## Getting Started
    To get started, run `pip install my-package`.
    ## Example
    ```python
    print("Hello, World!")
    ```
    """
    full_text = text.replace("('word ' * 500)", "word " * 500)
    d = write_readme(tmp_path, full_text)
    score, _ = metric({"local_dir": d})
    # Expected: 0.4 (length) + 0.35 (install) + 0.25 (code) = 1.0
    assert score == 1.0

def test_install_section_variations(tmp_path):
    """Covers various keywords for detecting an installation section."""
    base_text = "word " * 60  # 0.1 base score
    # Test with a header like 'Setup'
    d = write_readme(tmp_path, f"# Setup\n{base_text}")
    score, _ = metric({"local_dir": d})
    assert score == pytest.approx(0.1 + 0.35)

    # Test with a phrase like 'requirements.txt'
    d = write_readme(tmp_path, f"See requirements.txt for details.\n{base_text}")
    score, _ = metric({"local_dir": d})
    assert score == pytest.approx(0.1 + 0.35)

def test_indented_code_snippet(tmp_path):
    """Covers indented code blocks (besides fenced ones)."""
    text = "An example:\n\n    def main():\n        print('hello')\n\n" + "word " * 60
    d = write_readme(tmp_path, text)
    score, _ = metric({"local_dir": d})
    # Expected: 0.25 (code) + 0.1 (length)
    assert score == pytest.approx(0.25 + 0.1)

def test_metric_graceful_failure(mocker):
    """Covers the main try/except block to ensure it returns 0.0 on error."""
    mocker.patch('src.metrics.ramp_up_time._read_local_readme', side_effect=Exception("mock error"))
    score, lat = metric({"local_dir": "/fake/dir"})
    assert score == 0.0
    assert lat >= 0

# --- 3. Tests for Remote README Fetching (using Mocking) ---

@pytest.fixture
def mock_requests_get(mocker):
    """A pytest fixture to mock the `requests.get` method."""
    return mocker.patch('requests.get')

def test_remote_fetch_requests_not_installed(mocker):
    """Covers the case where 'requests' is not installed, returning a score of 0."""
    mocker.patch.dict('sys.modules', {'requests': None})
    score, _ = metric({"url": "https://github.com/owner/repo"})
    assert score == 0.0

def test_remote_fetch_github_success(mock_requests_get):
    """Covers a successful README fetch from a GitHub URL."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "word " * 250  # Score 0.25
    mock_requests_get.return_value = mock_response

    score, _ = metric({"url": "https://github.com/owner/repo"})
    assert score == 0.25
    mock_requests_get.assert_called_with(
        "https://raw.githubusercontent.com/owner/repo/main/README.md",
        timeout=6.0
    )

def test_remote_fetch_github_master_fallback(mock_requests_get):
    """Covers the fallback from the 'main' to 'master' branch for GitHub."""
    mock_response_fail = MagicMock(status_code=404)
    mock_response_success = MagicMock(status_code=200, text="# Setup\n" + "word "*50)
    mock_requests_get.side_effect = [mock_response_fail, mock_response_success]

    score, _ = metric({"url": "https://github.com/owner/repo"})
    # Expected: 0.1 (length) + 0.35 (install)
    assert score == pytest.approx(0.1 + 0.35)
    assert mock_requests_get.call_count == 2

def test_remote_fetch_all_fails(mock_requests_get):
    """Covers the case where all remote fetch attempts fail due to errors."""
    mock_requests_get.side_effect = Exception("Network Error")
    score, _ = metric({"url": "https://github.com/owner/repo"})
    assert score == 0.0
    
