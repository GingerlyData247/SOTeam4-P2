# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# METRIC: performance_claims
# REQUIREMENTS SATISFIED: performance_claims metric score

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/metrics/performance_claims.py

Computes the Performance Claims metric for a model artifact.

This metric estimates how credible and established a model’s claimed
performance is by analyzing its visibility and adoption within the
Hugging Face ecosystem. As a proxy for real-world validation, the metric
primarily uses the model’s download count to infer how widely the model
has been evaluated and trusted by the community.

Scoring approach:
    - Download counts are mapped to a tiered score between 0.0 and 1.0
    - Models with higher adoption receive higher performance-claims scores
    - Models that cannot be retrieved or do not exist receive a score of 0.0

The metric is lightweight, Hugging Face–only, and fault-tolerant, always
returning a bounded score and measured latency per the Phase 2
specification.
"""
from __future__ import annotations
from huggingface_hub.utils import HfHubHTTPError

HfHubHTTPError = HfHubHTTPError

# Imports from standard Python libraries
import time
import logging
from typing import Any, Dict, Tuple

# Import the specific functions we need from the huggingface_hub library
from huggingface_hub import model_info

# This gets the same logger instance used by run.py, so all logs go to the same place.
logger = logging.getLogger("phase1_cli")


# The 'metric' function is the required entry point for all metric files.
# It takes a 'resource' dictionary (containing URL, name, etc.)
# and must return a tuple containing a float (the score) and an int (the latency).
def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculates a score based on the model's popularity on the Hugging Face Hub,
    primarily using its download count.
    """
    # Start a timer to measure how long this metric takes to run.
    start_time = time.perf_counter()
    # Initialize the score to 0.0. It will be updated only on success.
    score = 0.0

    # Use a try...except block to handle potential errors, like network issues or if the model doesn't exist.
    try:
        # Extract the repository ID (e.g., "google/gemma-2b") from the resource dictionary.
        # This name is parsed from the URL by the main run.py script.
        repo_id = resource.get("name")
        if not repo_id:
            raise ValueError("Resource dictionary is missing a 'name' key for the repo_id.")

        # This is the main API call. It fetches all metadata for the given model repo_id.
        info = model_info(repo_id)
        # Safely get the download count. If it's not available, default to 0.
        downloads = info.downloads or 0
        # This log message is useful for debugging. It will only show up if LOG_LEVEL is 2.
        logger.debug(f"Successfully fetched info for {repo_id}. Downloads: {downloads}")
        
        # --- Scoring Logic ---
        # Convert the raw download number into a score between 0.0 and 1.0.
        # This simple tiered system assigns higher scores to more popular models.
        if downloads > 1_000_000:
            score = 1.0
        elif downloads > 100_000:
            score = 0.8
        elif downloads > 10_000:
            score = 0.6
        elif downloads > 1_000:
            score = 0.4
        elif downloads > 100:
            score = 0.2
        else:
            score = 0.1

    except HfHubHTTPError:
        # This specific error is for when the Hugging Face Hub returns an HTTP error (e.g., 404 Not Found).
        # In this case, the model doesn't exist or is private, so it gets a score of 0.
        logger.error(f"Could not find model '{repo_id}' on the Hub.")
        score = 0.0
    except Exception as e:
        # This is a general catch-all for any other unexpected errors (e.g., no internet connection).
        logger.exception(f"An unexpected error occurred while fetching info for {repo_id}: {e}")
        score = 0.0

    # Stop the timer.
    end_time = time.perf_counter()
    # Calculate the total time taken in milliseconds.
    latency_ms = int((end_time - start_time) * 1000)
    
    # Return the final score and the calculated latency.
    return score, latency_ms
