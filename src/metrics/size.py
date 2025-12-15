# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# METRIC: size
# REQUIREMENTS SATISFIED: size metric score

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/metrics/size.py

Computes the Size Compatibility metric for a model artifact.

This metric evaluates how suitable a model is for deployment across
different hardware environments by analyzing the total storage size of
the model repository. Size information is retrieved from the Hugging Face
API and mapped to compatibility scores for several common deployment
targets.

Hardware targets evaluated:
    - Raspberry Pi
    - Jetson Nano
    - Desktop PC
    - AWS server-class instances

Scoring approach:
    - Model size is determined using Hugging Face metadata (usedStorage
      or summed file sizes as a fallback)
    - Size is normalized against hardware-specific capacity thresholds
    - Smaller models receive higher compatibility scores

The metric is deterministic, fault-tolerant, and always returns a
dictionary of bounded scores along with measured latency per the Phase 2
specification.
"""
# src/metrics/size.py
from __future__ import annotations
import time
import logging
from typing import Any, Dict, Tuple
# Import HfApi client
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token, RepositoryNotFoundError

logger = logging.getLogger("phase1_cli")

HARDWARE_MAX_GB = {
    "raspberry_pi": 1.0,
    "jetson_nano": 2.0,
    "desktop_pc": 6.0,
    "aws_server": 10.0,
}

def _normalize_size_score(size_gb: float, max_gb: float) -> float:
    """Linearly scales size into a [0, 1] score, where smaller is better."""
    if not isinstance(size_gb, (int, float)) or not isinstance(max_gb, (int, float)):
         logger.warning(f"Invalid input to _normalize_size_score: size={size_gb}, max={max_gb}")
         return 0.0
    if size_gb <= 0: return 1.0
    if max_gb <= 0: return 0.0 # Avoid division by zero
    if size_gb >= max_gb: return 0.0
    score = 1.0 - (size_gb / max_gb)
    # Add logging inside normalization
    # logger.debug(f"Normalizing size: {size_gb:.3f} GB against max {max_gb:.1f} GB -> Score {score:.2f}") # Keep log cleaner
    return score

def metric(resource: Dict[str, Any]) -> Tuple[Dict[str, float], int]:
    """
    Fetches model repository size using HfApi.model_info() -> usedStorage
    and returns hardware scores. Corrected score assignment and return.
    """
    start_time = time.perf_counter()
    model_repo_id = resource.get("name")
    # Initialize scores to 0.0
    scores: Dict[str, float] = {key: 0.0 for key in HARDWARE_MAX_GB}
    total_size_bytes = 0 # Initialize size

    if not model_repo_id:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return scores, latency_ms

    try:
        token = get_token()
        if not token:
            logger.warning("Hugging Face token not found by size metric. API calls may fail or be rate-limited.")
        else:
             logger.debug(f"Size metric using detected Hugging Face token for '{model_repo_id}'.")

        # --- Instantiate HfApi client ---
        api = HfApi(token=token)

        # --- Attempt to get size using model_info and the 'usedStorage' attribute ---
        try:
            logger.debug(f"Attempting api.model_info() for '{model_repo_id}' to get 'usedStorage'")
            m_info = api.model_info(model_repo_id)

            # Primary method: Check for usedStorage
            if hasattr(m_info, 'usedStorage') and m_info.usedStorage is not None and m_info.usedStorage > 0:
                 total_size_bytes = m_info.usedStorage
                 logger.debug(f"Using model_info().usedStorage for '{model_repo_id}': {total_size_bytes} bytes")
            # Fallback: Sum siblings from model_info if usedStorage is missing/zero
            elif hasattr(m_info, 'siblings') and m_info.siblings:
                 logger.debug(f"'usedStorage' not available/zero for '{model_repo_id}'. Falling back to summing siblings from model_info.")
                 valid_siblings = [sf for sf in m_info.siblings if hasattr(sf, 'size') and sf.size is not None]
                 if valid_siblings:
                      total_size_bytes = sum(sf.size for sf in valid_siblings)
                      logger.debug(f"Summed size from {len(valid_siblings)} model_info siblings for '{model_repo_id}': {total_size_bytes} bytes")

        except RepositoryNotFoundError:
             logger.error(f"Size metric: Repository '{model_repo_id}' (type 'model') not found on Hub.")
        except Exception as e:
            logger.error(f"Size metric: Error during model_info call for '{model_repo_id}': {e}", exc_info=False)


        # --- Calculate scores based on the determined size ---
        calculated_scores: Dict[str, float] = {} # Use a temporary dict
        if total_size_bytes > 0:
            size_gb = total_size_bytes / (1024 ** 3)
            logger.debug(f"Calculated size for '{model_repo_id}': {size_gb:.3f} GB")
            for hardware, max_gb in HARDWARE_MAX_GB.items():
                # Calculate and store rounded score directly
                calculated_scores[hardware] = round(_normalize_size_score(size_gb, max_gb), 2) 
            logger.debug(f"Calculated scores dictionary for '{model_repo_id}': {calculated_scores}")
            scores = calculated_scores # Assign the calculated scores
        else:
             if model_repo_id:
                logger.warning(f"Final calculated size for '{model_repo_id}' is zero or could not be determined. Scores remain {scores}.")

    except Exception as e:
        logger.error(f"Size metric: An unexpected error occurred while processing '{model_repo_id}': {e}", exc_info=False)

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    # Log the exact dictionary being returned
    logger.debug(f"Returning final scores for '{model_repo_id}': {scores}") 
    return scores, latency_ms # Return the scores dictionary directly
