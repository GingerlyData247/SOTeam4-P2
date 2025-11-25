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


# Required in some Pydantic v2 setups when using __future__.annotations + generics.
ModelCreate.model_rebuild()
ModelUpdate.model_rebuild()
ModelOut.model_rebuild()
Page.model_rebuild()
