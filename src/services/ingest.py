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
