"""Request/response schemas and status vocabularies.

The job view produced by :meth:`inference_platform.store.JobStore.job_view`
deliberately matches what the AIMViewer front end already consumes, so the
Django layer can pass it through untouched.
"""
from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TERMINAL = (DONE, FAILED, CANCELLED)


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TERMINAL = (SUCCEEDED, FAILED, CANCELLED)


class Owner(BaseModel):
    user_id: Optional[int] = None
    username: str = ""

    @property
    def key(self) -> Optional[str]:
        if self.user_id is not None:
            return f"user:{self.user_id}"
        if self.username:
            return f"name:{self.username}"
        return None


class RegionSpec(BaseModel):
    """One annotated region, in level-0 slide coordinates."""

    key: str = Field(min_length=1, max_length=200)
    region_id: Optional[Union[int, str]] = None
    label_slug: str = ""
    roi_bbox: list[float] = Field(min_length=4, max_length=4)
    polygon: list[Any] = Field(default_factory=list)

    @field_validator("roi_bbox")
    @classmethod
    def _bbox_positive(cls, value: list[float]) -> list[float]:
        if value[2] <= 0 or value[3] <= 0:
            raise ValueError("roi_bbox width and height must be positive")
        return value


class JobSubmission(BaseModel):
    """A batch of regions to run through one model on one slide."""

    model_name: str = Field(min_length=1, max_length=64)
    slide_path: str = Field(min_length=1)
    regions: list[RegionSpec] = Field(min_length=1)
    downsample: float = Field(default=1.0, ge=1.0)
    overlay_downsample: Optional[float] = Field(default=None, ge=1.0)
    hyperparameters: Optional[dict[str, Any]] = None
    image_id: Optional[int] = None
    owner: Optional[Owner] = None
    priority: int = Field(default=0, ge=0, le=9)
    client_ref: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _unique_keys(self) -> "JobSubmission":
        seen: set[str] = set()
        for region in self.regions:
            if region.key in seen:
                raise ValueError(f"duplicate region key '{region.key}'")
            seen.add(region.key)
        return self


class CancelResponse(BaseModel):
    job: dict[str, Any]


class ModelDescriptor(BaseModel):
    """What a worker publishes about the model it serves."""

    name: str
    version: str = ""
    description: str = ""
    requires_mask: bool = False
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    supported_formats: list[str] = Field(default_factory=list)
    pool: str = "general"
    image: Optional[str] = None
    published_at: Optional[str] = None
