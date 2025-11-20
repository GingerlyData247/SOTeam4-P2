from __future__ import annotations
from typing import Dict, Any
from .scoring import ScoringService, NON_LATENCY
from .registry import RegistryService
from ..schemas.models import ModelCreate, ModelOut
from src.metrics.treescore import extract_parents_from_resource

class IngestService:
    def __init__(self, registry: RegistryService):
        self._registry = registry
        self._scoring = ScoringService()

    def ingest_hf(self, name_or_url: str) -> ModelOut:
        # Normalize into resource dict expected by metrics
        name = name_or_url.split("huggingface.co/")[-1] if "huggingface.co" in name_or_url else name_or_url
        resource = {"name": name, "url": f"https://huggingface.co/{name}"}
        scores = self._scoring.rate(resource)

        # Gate: each NON-LATENCY metric must be >= 0.5 before we ingest
        for metric_name in NON_LATENCY:
            if scores["subs"].get(metric_name, 0.0) < 0.5:
                raise ValueError(f"Ingest rejected: {metric_name}={scores['subs'].get(metric_name):.2f} < 0.50")
        
        # Use the enriched `resource` to derive lineage parents for this model
        parents = extract_parents_from_resource(resource)
    
        # If accepted, create minimal registry record (artifacts can be uploaded later)
        mc = ModelCreate(
            name=name, 
            version="1.0.0", 
            card="", 
            tags=["ingested", "hf"], 
            source_uri=resource["url"],
            metadata={"parents": parents},      # <- lineage data stored here
        )
        return self._registry.create(mc)