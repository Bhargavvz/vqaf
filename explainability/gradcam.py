"""
Grad-CAM Explainability Module
================================
Gradient-weighted Class Activation Mapping for visual explanations
of the Vision-Language model's predictions.

Generates heatmaps showing which image regions influenced the answer.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger


class GradCAM:
    """
    Grad-CAM implementation for Vision-Language models.
    
    Hooks into the last convolutional/attention layer of the vision
    encoder to compute gradient-weighted activation maps.
    
    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(image, question, answer_logits)
    """
    
    def __init__(self, model, target_layer: Optional[str] = None):
        """
        Args:
            model: The VLM model (Qwen3-VL).
            target_layer: Name of the target layer for Grad-CAM.
                         If None, auto-detects the last vision layer.
        """
        self.model = model
        self.target_layer = target_layer
        
        self.activations = None
        self.gradients = None
        self._hooks = []
        
        # Find and hook the target layer
        self._setup_hooks()
    
    def _find_visual_layer(self):
        """
        Auto-detect the last visual encoder layer.
        
        Searches for common vision encoder layer names in the model.
        """
        target = None
        
        # Search for visual encoder layers
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in [
                "visual", "vision", "vit", "patch_embed",
                "encoder.layers", "blocks"
            ]):
                target = module
                self.target_layer = name
        
        if target is None:
            # Fallback: use the last layer with 2D outputs
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.LayerNorm)):
                    target = module
                    self.target_layer = name
        
        return target
    
    def _setup_hooks(self):
        """Register forward and backward hooks on the target layer."""
        if self.target_layer:
            target = dict(self.model.named_modules()).get(self.target_layer)
        else:
            target = self._find_visual_layer()
        
        if target is None:
            logger.warning("Could not find target layer for Grad-CAM")
            return
        
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))
        
        logger.debug(f"Grad-CAM hooks registered on: {self.target_layer}")
    
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class_idx: Optional[int] = None,
        image_size: Tuple[int, int] = (448, 448),
    ) -> Optional[np.ndarray]:
        """
        Generate Grad-CAM heatmap for the given inputs.
        
        Args:
            inputs: Model inputs (from processor).
            target_class_idx: Target token index for backprop.
                            If None, uses the predicted token.
            image_size: Size to resize the heatmap to.
        
        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1],
            or None if generation fails.
        """
        if not self._hooks:
            logger.warning("No hooks registered. Cannot generate Grad-CAM.")
            return None
        
        try:
            # Enable gradients temporarily
            self.model.eval()
            
            # Ensure inputs require grad
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    val.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get target
            if target_class_idx is None:
                target_class_idx = logits[:, -1, :].argmax(dim=-1)
            
            # Backward pass
            self.model.zero_grad()
            target_logit = logits[:, -1, target_class_idx]
            if target_logit.dim() > 0:
                target_logit = target_logit.sum()
            target_logit.backward(retain_graph=True)
            
            # Compute Grad-CAM
            if self.activations is None or self.gradients is None:
                logger.warning("No activations/gradients captured.")
                return None
            
            # Global average pooling of gradients
            weights = self.gradients.mean(dim=-1, keepdim=True)
            if weights.dim() > self.activations.dim():
                weights = weights.squeeze()
            
            # Weighted combination of activations
            cam = (weights * self.activations).sum(dim=-1)
            
            # ReLU (only positive influences)
            cam = F.relu(cam)
            
            # Normalize
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Reshape to 2D if needed
            cam_np = cam.detach().cpu().numpy()
            if cam_np.ndim == 1:
                # For ViT: reshape sequence to square grid
                seq_len = cam_np.shape[0]
                side = int(np.sqrt(seq_len))
                if side * side == seq_len:
                    cam_np = cam_np.reshape(side, side)
                else:
                    cam_np = cam_np[:side * side].reshape(side, side)
            elif cam_np.ndim == 3:
                cam_np = cam_np[0]  # Take first batch
            
            # Resize to image dimensions
            from PIL import Image as PILImage
            cam_pil = PILImage.fromarray((cam_np * 255).astype(np.uint8))
            cam_pil = cam_pil.resize(image_size, PILImage.BILINEAR)
            heatmap = np.array(cam_pil).astype(np.float32) / 255.0
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return None
    
    def create_overlay(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Create a heatmap overlay on the original image.
        
        Args:
            image: Original PIL Image.
            heatmap: Grad-CAM heatmap (H, W) in [0, 1].
            alpha: Overlay transparency.
            colormap: Matplotlib colormap name.
        
        Returns:
            PIL Image with heatmap overlay.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Resize heatmap to match image
        heatmap_pil = Image.fromarray(heatmap_colored)
        heatmap_pil = heatmap_pil.resize(image.size, Image.BILINEAR)
        
        # Blend
        image_array = np.array(image.convert("RGB"))
        heatmap_array = np.array(heatmap_pil)
        
        overlay = (alpha * heatmap_array + (1 - alpha) * image_array).astype(np.uint8)
        
        return Image.fromarray(overlay)
    
    def cleanup(self):
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        self.cleanup()
