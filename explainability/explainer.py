"""
Explainability Engine
======================
Combines Grad-CAM, attention visualization, and text explanation
into a unified explainability pipeline for Medical VQA.

Produces structured output with answer, confidence, explanation,
and visual heatmap.
"""

import base64
import io
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger

from medical_vqa.explainability.gradcam import GradCAM
from medical_vqa.explainability.attention_viz import AttentionVisualizer


class ExplainabilityEngine:
    """
    Unified explainability engine that combines:
    1. Grad-CAM visual explanations
    2. Attention map visualization
    3. Text explanation generation
    
    Produces a structured output with all explanation modalities.
    """
    
    def __init__(
        self,
        model,
        processor,
        config: Dict[str, Any],
    ):
        """
        Args:
            model: The VQA model.
            processor: The input processor.
            config: Explainability configuration.
        """
        self.model = model
        self.processor = processor
        self.config = config
        
        # Initialize Grad-CAM
        gradcam_config = config.get("gradcam", {})
        self.gradcam = None
        if gradcam_config.get("enabled", True):
            try:
                target_layer = gradcam_config.get("target_layer")
                self.gradcam = GradCAM(model, target_layer)
                logger.info("Grad-CAM initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Grad-CAM: {e}")
        
        # Initialize Attention Visualizer
        attn_config = config.get("attention_maps", {})
        self.attn_viz = None
        if attn_config.get("enabled", True):
            try:
                self.attn_viz = AttentionVisualizer(
                    model,
                    num_heads_to_visualize=attn_config.get("num_heads_to_visualize", 4)
                )
                logger.info("Attention visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize attention visualizer: {e}")
    
    def explain(
        self,
        image: Image.Image,
        question: str,
        answer: str,
        model_outputs: Dict[str, Any],
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for a VQA prediction.
        
        Args:
            image: Input medical image.
            question: The question asked.
            answer: The model's answer.
            model_outputs: Raw model outputs (logits, etc.).
            save_dir: Optional directory to save visualizations.
        
        Returns:
            Structured explanation dict with:
                - answer: The predicted answer
                - confidence: Prediction confidence score
                - explanation: Text explanation
                - visual_heatmap: Base64-encoded heatmap image (or path)
                - attention_map: Base64-encoded attention visualization
        """
        result = {
            "answer": answer,
            "confidence": 0.0,
            "explanation": "",
            "visual_heatmap": None,
            "attention_map": None,
        }
        
        # Compute confidence from logits
        if "logits" in model_outputs:
            result["confidence"] = self._compute_confidence(model_outputs["logits"])
        elif "output_tokens" in model_outputs:
            # Use a heuristic confidence
            result["confidence"] = 0.85  # Default for generation models
        
        # Extract text explanation from model output
        if "explanation" in model_outputs and model_outputs["explanation"]:
            result["explanation"] = model_outputs["explanation"]
        elif "full_response" in model_outputs:
            result["explanation"] = self._extract_explanation(
                model_outputs["full_response"], answer
            )
        
        # Generate Grad-CAM heatmap
        if self.gradcam:
            heatmap = self._generate_gradcam(image, model_outputs, save_dir)
            if heatmap is not None:
                result["visual_heatmap"] = heatmap
        
        # Generate attention visualization
        if self.attn_viz and "inputs" in model_outputs:
            attn_map = self._generate_attention_map(
                image, model_outputs["inputs"], save_dir
            )
            if attn_map is not None:
                result["attention_map"] = attn_map
        
        return result
    
    def _compute_confidence(self, logits: torch.Tensor) -> float:
        """Compute prediction confidence from logits."""
        try:
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            confidence = probs.max().item()
            return round(confidence, 4)
        except Exception:
            return 0.0
    
    def _extract_explanation(self, full_response: str, answer: str) -> str:
        """Extract explanation from the full model response."""
        response = full_response.strip()
        
        # Remove the answer part to get the explanation
        if answer and response.startswith(answer):
            explanation = response[len(answer):].strip()
            # Remove leading separators
            for sep in [".", ":", "-", ","]:
                explanation = explanation.lstrip(sep).strip()
            return explanation
        
        # Try to find explanation after common delimiters
        for delimiter in ["Explanation:", "Because ", "This is because", "The reason"]:
            if delimiter in response:
                idx = response.index(delimiter)
                return response[idx:].strip()
        
        # If response is longer than the answer, the extra part is the explanation
        if len(response) > len(answer) + 10:
            return response
        
        return "No detailed explanation available."
    
    def _generate_gradcam(
        self,
        image: Image.Image,
        model_outputs: Dict,
        save_dir: Optional[str] = None,
    ) -> Optional[str]:
        """Generate Grad-CAM heatmap and return as base64 or file path."""
        try:
            if "inputs" not in model_outputs:
                return None
            
            heatmap = self.gradcam.generate(
                inputs=model_outputs["inputs"],
                image_size=image.size[::-1],  # (H, W)
            )
            
            if heatmap is None:
                return None
            
            # Create overlay
            overlay = self.gradcam.create_overlay(image, heatmap)
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, "gradcam_heatmap.png")
                overlay.save(path)
                return path
            else:
                return self._image_to_base64(overlay)
                
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return None
    
    def _generate_attention_map(
        self,
        image: Image.Image,
        inputs: Dict[str, torch.Tensor],
        save_dir: Optional[str] = None,
    ) -> Optional[str]:
        """Generate attention map visualization."""
        try:
            attention_maps = self.attn_viz.extract_attention(inputs)
            
            if not attention_maps:
                return None
            
            # Use the last layer's attention
            layer_name, weights = attention_maps[-1]
            
            viz = self.attn_viz.create_multi_head_visualization(
                image, weights,
                save_path=os.path.join(save_dir, "attention_map.png") if save_dir else None
            )
            
            if viz and not save_dir:
                return self._image_to_base64(viz)
            elif save_dir:
                return os.path.join(save_dir, "attention_map.png")
            
            return None
            
        except Exception as e:
            logger.error(f"Attention visualization failed: {e}")
            return None
    
    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    
    def cleanup(self):
        """Clean up hooks and resources."""
        if self.gradcam:
            self.gradcam.cleanup()
        if self.attn_viz:
            self.attn_viz.cleanup_hooks()
