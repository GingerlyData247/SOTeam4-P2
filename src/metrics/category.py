# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# METRIC: category
# REQUIREMENTS SATISFIED: artifact categorization support

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/metrics/category.py

Determines a high-level category for an artifact based on its source.

This metric assigns a descriptive category to a model or code artifact
using available metadata. For Hugging Face models, the category is
derived from the model’s pipeline_tag when available. For GitHub-based
artifacts, the category defaults to a code repository classification.

The metric is designed to be lightweight, fast, and fault-tolerant,
providing a best-effort classification while always returning a valid
fallback value and measured latency.
"""
import time
from typing import Any, Dict, Tuple
from huggingface_hub import model_info
from huggingface_hub.utils import HfHubHTTPError

def metric(resource: Dict[str, Any]) -> Tuple[str, int]:
    """
    Determines a specific category for a model, using the Hugging Face
    pipeline_tag if available.
    """
    start_time = time.perf_counter()
    category = "Model" # Default fallback category

    if "huggingface.co" in resource['url']:
        try:
            info = model_info(resource['name'])
            if info.pipeline_tag:
                category = info.pipeline_tag
        except HfHubHTTPError:
            category = "Model (Not Found)"
    elif "github.com" in resource['url']:
        category = "Code Repository"

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return category, latency_ms
