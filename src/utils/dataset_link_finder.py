# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: DATASET LINK DISCOVERY UTILITIES
# REQUIREMENTS SATISFIED: dataset discovery support for scoring and ingestion

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/utils/dataset_link_finder.py

Provides utilities for discovering dataset references associated with a
Hugging Face model.

This module implements robust logic to identify dataset links and names
connected to a model artifact. It prioritizes structured Hugging Face
metadata when available and falls back to carefully constrained README
parsing strategies to avoid false positives.

Key responsibilities:
    - Extract dataset references from Hugging Face model metadata
    - Parse local or remote README files for dataset declarations
    - Normalize dataset URLs and names into canonical HF dataset IDs
    - Support YAML frontmatter, HTML links, and contextual text mentions
    - Return deterministic, de-duplicated results with measured latency

These utilities are used by dataset-related metrics and ingestion logic
to ensure dataset associations are identified accurately and safely in
accordance with the Phase 2 specification.
"""
# src/utils/dataset_link_finder.py
from __future__ import annotations
import re
import os
import time
import logging
import yaml
from typing import List, Tuple, Optional, Dict, Set
from html.parser import HTMLParser
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download, model_info, get_token
from huggingface_hub.utils import RepositoryNotFoundError

logger = logging.getLogger("phase1_cli")

# --- Constants and Regexes ---

# Regex for valid HF dataset names: owner/name or single word (like 'squad')
# Allows letters, numbers, '.', '_', '-'
# Owner part is optional for canonical datasets
# Corrected: Ensure it matches the whole string and handles optional owner part correctly
DATASET_NAME_REGEX = re.compile(r"^(?:[a-zA-Z0-9\._-]+/)?([a-zA-Z0-9\._-]+)$")


# Regex to find potential dataset mentions (owner/name) in text near 'dataset' keywords
DATASET_MENTION_REGEX = re.compile(r'\b([a-zA-Z0-9\._-]+/[a-zA-Z0-9\._-]+)\b')

# Keywords indicating a nearby mention is likely a dataset
DATASET_CONTEXT_KEYWORDS = {'dataset', 'datasets', 'trained on', 'trained with', 'fine-tuned on', 'data source'}

# --- Helper Classes and Functions ---

class HrefParser(HTMLParser):
    """Extracts href attributes from HTML anchor tags."""
    def __init__(self):
        super().__init__()
        self.hrefs: List[str] = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            # Ensure value is treated as string
            self.hrefs.extend(str(value) for name, value in attrs if name == 'href' and value)

def _read_local_readme(local_dir: str) -> Optional[str]:
    """Reads the first found README file in a local directory."""
    if not local_dir or not os.path.isdir(local_dir):
        return None
    for fname in ("README.md", "README.rst", "README.txt", "README"):
        path = os.path.join(local_dir, fname)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            except Exception as e:
                logger.debug(f"Could not read local README {path}: {e}")
    return None

def _fetch_readme_from_hf(repo_id: str, token: str | None) -> Optional[str]:
    """Downloads the README.md file for a given model repo_id."""
    if not repo_id: return None
    try:
        readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="model",
            token=token,
            library_name="phase1-cli",
            etag_timeout=10,
        )
        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except RepositoryNotFoundError:
        logger.debug(f"README fetch: Model repo '{repo_id}' not found.")
    except Exception as e:
        logger.debug(f"Could not fetch README for '{repo_id}' from Hub: {e}")
    return None

def _normalize_dataset_ref(ref: str) -> Optional[str]:
    """
    Normalizes a potential dataset reference (URL or name) into a canonical
    HF dataset ID ('owner/name' or 'name') if valid. Returns None otherwise.
    """
    if not ref or not isinstance(ref, str): return None # Added type check
    ref = ref.strip().lower()

    # Case 1: Full HF Dataset URL
    if "huggingface.co/datasets/" in ref:
        try:
            path_parts = urlparse(ref).path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[0] == "datasets":
                # Join only the owner/name or just name
                dataset_id = "/".join(path_parts[1:3] if len(path_parts) > 2 else path_parts[1:2]) 
                # Validate the extracted ID format
                if DATASET_NAME_REGEX.match(dataset_id):
                    return dataset_id
        except Exception: pass

    # Case 2: Direct name ('owner/name' or 'name')
    if DATASET_NAME_REGEX.match(ref):
        return ref

    return None

# --- Public Function ---

def find_datasets_from_resource(resource: Dict) -> Tuple[List[str], int]:
    """
    Finds dataset references associated with a model resource.
    """
    start_time = time.perf_counter()
    repo_id = resource.get("name")
    local_dir = resource.get("local_dir")
    token = get_token()

    found_refs: Set[str] = set()

    # Priority 1: Check declared datasets in model metadata
    if repo_id:
        try:
            info = model_info(repo_id, token=token)
            # Ensure info.datasets exists and is iterable
            if hasattr(info, 'datasets') and info.datasets is not None: 
                # Check it's actually a list before iterating
                if isinstance(info.datasets, list):
                    for ds in info.datasets:
                        if isinstance(ds, str) and (norm_ref := _normalize_dataset_ref(ds)):
                            found_refs.add(norm_ref)
                    if found_refs:
                        logger.debug(f"Found declared datasets in metadata for '{repo_id}': {found_refs}")
                else:
                     logger.debug(f"Metadata 'datasets' field for '{repo_id}' is not a list: {type(info.datasets)}")
            else:
                 logger.debug(f"Metadata for '{repo_id}' does not have a 'datasets' attribute or it is None.")
        except Exception as e:
            logger.debug(f"Could not check model info for '{repo_id}' for datasets tag: {e}", exc_info=False)


    # Priority 2: Parse README (local or fetched)
    readme_text = _read_local_readme(local_dir)
    if not readme_text and repo_id: # Fetch if local read failed or wasn't possible
        readme_text = _fetch_readme_from_hf(repo_id, token)

    if readme_text:
        # Strategy A: Look for YAML frontmatter
        try: # Wrap YAML parsing in try-except
            if readme_text.strip().startswith("---"):
                parts = readme_text.split("---", 2)
                if len(parts) > 2:
                    metadata = yaml.safe_load(parts[1]) or {} # Ensure metadata is dict
                    datasets_yaml = metadata.get("datasets", [])
                    if isinstance(datasets_yaml, list): # Check type again
                        for item in datasets_yaml:
                            if isinstance(item, str) and (norm_ref := _normalize_dataset_ref(item)):
                                found_refs.add(norm_ref)
        except yaml.YAMLError as e:
             logger.debug(f"YAML parsing error in README for '{repo_id}': {e}")
        except Exception as e:
             logger.error(f"Unexpected error during YAML parsing for '{repo_id}': {e}", exc_info=False)

        # Strategy B: Find URLs pointing to HF Datasets
        urls_in_readme: List[str] = []
        try:
            parser = HrefParser(); parser.feed(readme_text); urls_in_readme.extend(parser.hrefs)
        except Exception as e:
             logger.debug(f"HTML parsing error in README for '{repo_id}': {e}")
        for url in urls_in_readme:
             # Check type before proceeding
             if isinstance(url, str) and "huggingface.co/datasets/" in url: 
                if norm_ref := _normalize_dataset_ref(url):
                     found_refs.add(norm_ref)

        # Strategy C: Find 'owner/name' mentions near context keywords (Lower priority)
        # Only do this if other methods failed to find anything
        if not found_refs: 
            try: # Wrap regex search in try-except
                for match in DATASET_MENTION_REGEX.finditer(readme_text):
                     mention = match.group(1)
                     context_start = max(0, match.start() - 50)
                     context_end = match.end() + 50
                     context = readme_text[context_start:context_end].lower()
                     if any(keyword in context for keyword in DATASET_CONTEXT_KEYWORDS):
                         if norm_ref := _normalize_dataset_ref(mention):
                             found_refs.add(norm_ref)
            except Exception as e:
                 logger.error(f"Error during regex mention search for '{repo_id}': {e}", exc_info=False)


    latency_ms = int((time.perf_counter() - start_time) * 1000)
    # Return sorted list for consistent output
    return sorted(list(found_refs)), latency_ms
