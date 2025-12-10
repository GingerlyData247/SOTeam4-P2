from __future__ import annotations
from typing import Optional, List, Dict, Any, Generic, TypeVar
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar("T")


class ModelCreate(BaseModel):
    name: str = Field(..., examples=["google-bert/bert-base-uncased"])
    version: str = Field(..., examples=["1.0.0"])
    card: str = Field("", description="Raw/markdown card text")
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    source_uri: Optional[str] = None


class ModelUpdate(BaseModel):
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ModelOut(BaseModel):
    id: str
    name: str
    version: str
    metadata: Dict[str, Any]


class Page(GenericModel, Generic[T]):
    items: List[T]
    next_cursor: Optional[str] = None


# ----------------------------------------------------------------------
# NEW: Rating models, matching OpenAPI ModelRating schema
# ----------------------------------------------------------------------


class SizeScore(BaseModel):
    raspberry_pi: float
    jetson_nano: float
    desktop_pc: float
    aws_server: float


class ModelRating(BaseModel):
    name: str
    category: str

    net_score: float
    net_score_latency: float

    ramp_up_time: float
    ramp_up_time_latency: float

    bus_factor: float
    bus_factor_latency: float

    performance_claims: float
    performance_claims_latency: float

    # NOTE: license is a numeric suitability score per spec, not a string
    license: float
    license_latency: float

    dataset_and_code_score: float
    dataset_and_code_score_latency: float

    dataset_quality: float
    dataset_quality_latency: float

    code_quality: float
    code_quality_latency: float

    reproducibility: float
    reproducibility_latency: float

    reviewedness: float
    reviewedness_latency: float

    tree_score: float
    tree_score_latency: float

    # Phase 2: size_score is an object with four floats
    size_score: SizeScore
    size_score_latency: float


# Required in some Pydantic v2 setups when using __future__.annotations + generics.
ModelCreate.model_rebuild()
ModelUpdate.model_rebuild()
ModelOut.model_rebuild()
Page.model_rebuild()
SizeScore.model_rebuild()
ModelRating.model_rebuild()
