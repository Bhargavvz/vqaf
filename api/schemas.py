"""
API Schemas
============
Pydantic models for request/response validation in the Medical VQA API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class VQARequest(BaseModel):
    """Request model for Medical VQA prediction."""
    question: str = Field(
        ...,
        description="Medical question about the uploaded image",
        min_length=3,
        max_length=500,
        examples=["Is there cardiomegaly?"]
    )
    include_explanation: bool = Field(
        default=True,
        description="Whether to include a clinical explanation"
    )
    include_heatmap: bool = Field(
        default=True,
        description="Whether to include a visual heatmap"
    )
    max_tokens: int = Field(
        default=256,
        ge=16,
        le=1024,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )


class VQAResponse(BaseModel):
    """Response model for Medical VQA prediction."""
    answer: str = Field(
        ...,
        description="Predicted answer to the medical question"
    )
    confidence: float = Field(
        ...,
        description="Model confidence score (0-1)"
    )
    explanation: str = Field(
        default="",
        description="Clinical explanation for the answer"
    )
    visual_heatmap: Optional[str] = Field(
        default=None,
        description="Base64-encoded heatmap image (PNG)"
    )
    knowledge_used: Optional[str] = Field(
        default=None,
        description="Retrieved medical knowledge used for the answer"
    )
    processing_time_ms: float = Field(
        default=0,
        description="Total processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    knowledge_index_ready: bool = Field(default=False)
    gpu_available: bool = Field(default=False)
    gpu_name: Optional[str] = Field(default=None)
    model_name: Optional[str] = Field(default=None)


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    error: str
    detail: Optional[str] = None
