# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: ARTIFACT INGESTION SERVICE
# REQUIREMENTS SATISFIED: Hugging Face model ingestion and scoring integration

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/services/ingest.py

Defines the ingestion service responsible for onboarding new model
artifacts into the registry.

This module implements Phase 2–compliant ingestion logic for Hugging Face
models. It coordinates normalization of model identifiers, metric
computation, lineage extraction, and persistence into the registry. The
service is intentionally lightweight and delegates scoring and storage
concerns to dedicated service layers.

Key responsibilities:
    - Normalize Hugging Face model identifiers
    - Build scoring-compatible resource representations
    - Invoke the scoring service to compute Phase 2 metrics
    - Extract parent/lineage information for models
    - Persist fully enriched artifacts into the registry

This service acts as a bridge between API routes and backend services,
ensuring that ingested artifacts are consistently scored, traceable,
and stored according to the Phase 2 specification.
"""
from __future__ import annotations
from typing import Dict, Any
from .scoring import ScoringService
from .registry import RegistryService
from ..schemas.models import ModelCreate, ModelOut
from src.utils.hf_normalize import normalize_hf_id
from src.metrics.treescore import extract_parents_from_resource


class IngestService:
    def __init__(self, registry: RegistryService):
        self._registry = registry
        self._scoring = ScoringService()

    def ingest_hf(self, *, final_name: str, url: str) -> ModelOut:
        """
        Phase 2–correct HF ingest:
        - trusts caller-provided name
        - no ingest gating
        - full scoring-compatible resource
        """

        hf_id = normalize_hf_id(url)

        resource: Dict[str, Any] = {
            "name": final_name,
            "url": f"https://huggingface.co/{hf_id}",
            "artifact_type": "model",
        }

        # Compute rating (no gate!)
        rating = self._scoring.rate(resource)

        # Extract lineage AFTER resource is fully built
        parents = extract_parents_from_resource(resource)

        mc = ModelCreate(
            name=final_name,
            version="1.0.0",
            card="",
            tags=["ingested", "hf"],
            source_uri=resource["url"],
            metadata={
                **rating,
                "parents": parents,
            },
        )

        return self._registry.create(mc)
