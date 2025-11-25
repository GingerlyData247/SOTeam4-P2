from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# -----------------------------------------
# SHARED SUB-SCHEMAS
# -----------------------------------------

class Score(BaseModel):
    """12 required rating attributes from autograder."""
    fairness: Optional[float] = None
    security: Optional[float] = None
    privacy: Optional[float] = None
    robustness: Optional[float] = None
    explainability: Optional[float] = None
    transparency: Optional[float] = None
    safety: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    reliability: Optional[float] = None
    usability: Optional[float] = None
    generalization: Optional[float] = None


class SizeInfo(BaseModel):
    """Required by Download URL + Cost + Metadata tests."""
    parameters: Optional[int] = None
    disk_size_bytes: Optional[int] = None


class Metadata(BaseModel):
    """Full metadata object required by Phase 2 autograder."""
    card: Optional[str] = None
    tags: Optional[List[str]] = None
    source_uri: Optional[str] = None
    license: Optional[str] = None
    parents: Optional[List[str]] = None
    download_url: Optional[str] = None
    score: Optional[Score] = None
    size: Optional[SizeInfo] = None
    cost: Optional[float] = None   # required for Artifact Cost Test


# -----------------------------------------
# INPUT SCHEMAS
# -----------------------------------------

class ModelCreate(BaseModel):
    """Used during INGEST / CREATE."""
    name: str
    version: str
    card: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Metadata] = None
    source_uri: Optional[str] = None


class ModelUpdate(BaseModel):
    """Used during PUT / UPDATE."""
    card: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Metadata] = None


# -----------------------------------------
# OUTPUT SCHEMA
# -----------------------------------------

class ModelOut(BaseModel):
    id: str
    name: str
    version: str
    metadata: Metadata


# -----------------------------------------
# PAGINATION
# -----------------------------------------

class Page(BaseModel):
    items: List[ModelOut]
    next_cursor: Optional[str] = None


# Force rebuild for forward refs
Score.model_rebuild()
SizeInfo.model_rebuild()
Metadata.model_rebuild()
ModelCreate.model_rebuild()
ModelUpdate.model_rebuild()
ModelOut.model_rebuild()
Page.model_rebuild()
