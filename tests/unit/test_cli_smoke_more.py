# ---------------------------------------------------------------------------
# Unit Tests: CLI URL Classification (Extended Cases)
#
# This test suite exercises additional and edge-case scenarios for the
# `classify_url` helper used by the CLI and metric execution pipeline.
#
# Covered behavior:
#   - Correct classification of Hugging Face model and dataset URLs
#   - Detection of code repositories hosted on GitHub and GitLab
#   - Robust handling of empty strings, whitespace, and case variations
#
# These tests help ensure stable URL classification behavior, which is
# foundational for correct artifact typing and metric selection in
# the Phase 1 / Phase 2 Trustworthy Model Registry.
# ---------------------------------------------------------------------------
import pytest
from run import classify_url

@pytest.mark.parametrize("u,cat", [
    ("https://huggingface.co/datasets/squad", "DATASET"),
    ("https://huggingface.co/bert-base-uncased", "MODEL"),
    ("https://github.com/pytorch/pytorch", "CODE"),
    ("https://gitlab.com/user/repo", "CODE"),
    ("", "CODE"),
    ("  https://HUGGINGFACE.CO/bert-base-uncased  ", "MODEL"),
])
def test_classify_url_more(u, cat):
    assert classify_url(u) == cat
