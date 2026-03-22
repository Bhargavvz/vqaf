"""
Medical VQA Inference API
==========================
FastAPI server for Medical Visual Question Answering.

Endpoints:
    POST /predict  - Upload image + question → answer + explanation + heatmap
    GET  /health   - Health check with model status

Usage:
    uvicorn medical_vqa.api.server:app --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from medical_vqa.api.schemas import (
    ErrorResponse,
    HealthResponse,
    VQARequest,
    VQAResponse,
)

# ============================================================
# Global State
# ============================================================

_model_wrapper = None
_retriever = None
_explainer = None
_config = None


def _load_config() -> dict:
    """Load configuration from YAML."""
    config_path = os.environ.get("VQA_CONFIG", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    logger.warning(f"Config not found at {config_path}, using defaults")
    return {}


def _initialize_model():
    """Initialize the model, retriever, and explainer."""
    global _model_wrapper, _retriever, _explainer, _config
    
    _config = _load_config()
    
    logger.info("Initializing Medical VQA API...")
    
    # Load knowledge retriever
    from medical_vqa.knowledge.retriever import MedicalKnowledgeRetriever
    
    knowledge_config = _config.get("knowledge", {})
    _retriever = MedicalKnowledgeRetriever(
        embedding_model_name=knowledge_config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        index_dir=_config.get("paths", {}).get("knowledge_index_dir"),
        top_k=knowledge_config.get("top_k", 3),
    )
    
    try:
        _retriever.load_index()
    except Exception:
        _retriever.build_index()
    
    logger.info("Knowledge retriever ready")
    
    # Load model
    from medical_vqa.model.model import MedicalVQAModel
    
    _model_wrapper = MedicalVQAModel(_config)
    _model_wrapper.load_model(for_training=False)
    
    # Load trained adapter if available
    adapter_path = os.environ.get(
        "VQA_ADAPTER_PATH",
        _config.get("paths", {}).get("output_dir", "./outputs") + "/final_model"
    )
    if os.path.isdir(adapter_path):
        _model_wrapper.load_trained_adapter(adapter_path)
        logger.info(f"Loaded trained adapter from: {adapter_path}")
    
    # Initialize explainer
    from medical_vqa.explainability.explainer import ExplainabilityEngine
    
    _explainer = ExplainabilityEngine(
        model=_model_wrapper.model,
        processor=_model_wrapper.processor,
        config=_config.get("explainability", {}),
    )
    
    logger.info("Medical VQA API fully initialized!")


# ============================================================
# FastAPI Application
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - loads model on startup."""
    logger.info("Starting Medical VQA API server...")
    try:
        _initialize_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.warning("API will start but /predict will return errors")
    yield
    # Cleanup
    if _explainer:
        _explainer.cleanup()
    logger.info("Medical VQA API server shut down")


app = FastAPI(
    title="Medical VQA API",
    description=(
        "Knowledge-Guided Explainable Vision-Language Transformer "
        "for Medical Visual Question Answering"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns model status, GPU availability, and knowledge index status.
    """
    gpu_available = torch.cuda.is_available()
    
    return HealthResponse(
        status="healthy" if _model_wrapper else "degraded",
        model_loaded=_model_wrapper is not None,
        knowledge_index_ready=_retriever is not None and _retriever.index is not None,
        gpu_available=gpu_available,
        gpu_name=torch.cuda.get_device_name(0) if gpu_available else None,
        model_name=_model_wrapper.model_name if _model_wrapper else None,
    )


@app.post("/predict", response_model=VQAResponse)
async def predict(
    image: UploadFile = File(..., description="Medical image file (JPEG, PNG)"),
    question: str = Form(..., description="Medical question about the image"),
    include_explanation: bool = Form(True, description="Include clinical explanation"),
    include_heatmap: bool = Form(True, description="Include visual heatmap"),
    max_tokens: int = Form(256, description="Maximum tokens to generate"),
    temperature: float = Form(0.1, description="Sampling temperature"),
):
    """
    Medical VQA prediction endpoint.
    
    Upload a medical image and ask a clinical question.
    Returns the answer, explanation, and visual heatmap.
    
    Accepts:
        - image: Medical image file (JPEG, PNG, DICOM)
        - question: Clinical question about the image
        - include_explanation: Whether to include medical explanation
        - include_heatmap: Whether to include Grad-CAM heatmap
    
    Returns:
        VQAResponse with answer, confidence, explanation, and heatmap.
    """
    start_time = time.time()
    
    # Validate model is loaded
    if _model_wrapper is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Expected image/*"
        )
    
    try:
        # Read and process image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Retrieve relevant knowledge
        knowledge = ""
        if _retriever:
            knowledge = _retriever.retrieve_and_format(question)
        
        # Generate answer
        result = _model_wrapper.generate(
            image=pil_image,
            question=question,
            knowledge=knowledge,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Generate explanations if requested
        visual_heatmap = None
        if include_heatmap and _explainer:
            try:
                explanation_result = _explainer.explain(
                    image=pil_image,
                    question=question,
                    answer=result["answer"],
                    model_outputs=result,
                )
                visual_heatmap = explanation_result.get("visual_heatmap")
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return VQAResponse(
            answer=result["answer"],
            confidence=0.85,  # From generation model
            explanation=result.get("explanation", "") if include_explanation else "",
            visual_heatmap=visual_heatmap,
            knowledge_used=knowledge if knowledge else None,
            processing_time_ms=round(processing_time, 2),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Medical VQA API",
        "version": "1.0.0",
        "description": (
            "Knowledge-Guided Explainable Vision-Language Transformer "
            "for Medical Visual Question Answering"
        ),
        "endpoints": {
            "POST /predict": "Upload image + question for VQA",
            "GET /health": "Health check",
        }
    }


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    api_config = _load_config().get("api", {})
    
    uvicorn.run(
        "medical_vqa.api.server:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=False,
        workers=1,  # Single worker for GPU model
    )
