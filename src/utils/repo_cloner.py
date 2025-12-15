# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: GIT REPOSITORY CLONING UTILITY
# REQUIREMENTS SATISFIED: Local repository inspection, reproducibility and code-quality metrics support

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/utils/repo_cloner.py

Utility functions for safely cloning external Git repositories into a temporary
local directory for analysis by downstream metrics (e.g., code quality,
reproducibility, dataset/code inspection).

Key Characteristics:
    - Uses shallow clones (depth=1) to minimize bandwidth, disk usage, and
      execution time.
    - Clones into a system-managed temporary directory to avoid polluting
      persistent storage.
    - Ensures cleanup on failure to prevent orphaned directories.

Failure Handling:
    - Any GitCommandError during cloning is caught and logged.
    - Temporary directories created during failed clone attempts are removed
      immediately.
    - Returns None on failure, allowing calling metrics to degrade gracefully.

Logging:
    - Uses the shared "phase1_cli" logger configured via src/utils/logging.py.
    - Emits INFO-level logs for clone attempts and success.
    - Emits ERROR-level logs on clone failure.

This module is intentionally minimal and defensive, as it operates on
untrusted external repositories supplied by users or extracted from metadata.
"""
import tempfile
import shutil
import logging
from git import Repo, GitCommandError

logger = logging.getLogger("phase1_cli")

def clone_repo_to_temp(repo_url: str) -> str | None:
    """
    Clones a Git repository to a temporary directory.
    Returns the path to the temp directory or None if it fails.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        logger.info(f"Cloning {repo_url} to {temp_dir}...")
        # Use depth=1 for a shallow clone to save time and space
        Repo.clone_from(repo_url, temp_dir, depth=1)
        logger.info(f"Successfully cloned {repo_url}.")
        return temp_dir
    except GitCommandError as e:
        logger.error(f"Failed to clone repository {repo_url}: {e}")
        shutil.rmtree(temp_dir)  # Clean up the failed clone
        return None
