"""
Attention Map Visualization Module
=====================================
Extracts and visualizes cross-attention maps from the
Vision-Language Transformer to show image-text alignment.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from loguru import logger


class AttentionVisualizer:
    """
    Extracts and visualizes attention maps from transformer layers.
    
    Shows which image patches attend to which text tokens,
    providing interpretable explanations of model decisions.
    """
    
    def __init__(self, model, num_heads_to_visualize: int = 4):
        """
        Args:
            model: The VLM model.
            num_heads_to_visualize: Number of attention heads to show.
        """
        self.model = model
        self.num_heads = num_heads_to_visualize
        self.attention_maps: List[torch.Tensor] = []
        self._hooks = []
    
    def register_hooks(self):
        """Register hooks to capture attention weights."""
        self.attention_maps.clear()
        
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and hasattr(module, "num_heads"):
                def hook_fn(module, input, output, name=name):
                    # Capture attention weights if returned
                    if isinstance(output, tuple) and len(output) > 1:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            self.attention_maps.append(
                                (name, attn_weights.detach().cpu())
                            )
                
                self._hooks.append(module.register_forward_hook(hook_fn))
        
        if self._hooks:
            logger.debug(f"Registered {len(self._hooks)} attention hooks")
        else:
            logger.warning("No attention layers found for visualization")
    
    def extract_attention(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Run a forward pass and extract attention maps.
        
        Args:
            inputs: Model inputs.
        
        Returns:
            List of (layer_name, attention_weights) tuples.
        """
        self.attention_maps.clear()
        self.register_hooks()
        
        with torch.no_grad():
            try:
                # Some models need output_attentions=True
                self.model(**inputs, output_attentions=True)
            except TypeError:
                self.model(**inputs)
        
        self.cleanup_hooks()
        
        results = [
            (name, weights.numpy())
            for name, weights in self.attention_maps
        ]
        
        logger.debug(f"Extracted attention from {len(results)} layers")
        return results
    
    def visualize_attention(
        self,
        image: Image.Image,
        attention_weights: np.ndarray,
        head_idx: int = 0,
        image_size: Tuple[int, int] = (448, 448),
    ) -> Image.Image:
        """
        Create an attention map visualization overlay on the image.
        
        Args:
            image: Original PIL Image.
            attention_weights: Attention weights array [heads, seq, seq].
            head_idx: Which attention head to visualize.
            image_size: Target image size.
        
        Returns:
            PIL Image with attention overlay.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Get attention for specific head
        if attention_weights.ndim == 4:
            attn = attention_weights[0, head_idx]  # [seq, seq]
        elif attention_weights.ndim == 3:
            attn = attention_weights[head_idx]  # [seq, seq]
        else:
            attn = attention_weights
        
        # Average across source tokens to get per-position importance
        attn_map = attn.mean(axis=0)  # [seq_len]
        
        # Try to reshape into 2D grid for image patches
        seq_len = len(attn_map)
        side = int(np.sqrt(seq_len))
        
        if side * side <= seq_len:
            attn_2d = attn_map[:side * side].reshape(side, side)
        else:
            attn_2d = attn_map.reshape(1, -1)
        
        # Normalize
        if attn_2d.max() > 0:
            attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min())
        
        # Apply colormap
        cmap = cm.get_cmap("viridis")
        attn_colored = cmap(attn_2d)[:, :, :3]
        attn_colored = (attn_colored * 255).astype(np.uint8)
        
        # Resize and overlay
        attn_pil = Image.fromarray(attn_colored)
        attn_pil = attn_pil.resize(image.size, Image.BILINEAR)
        
        # Blend
        image_array = np.array(image.convert("RGB"))
        attn_array = np.array(attn_pil)
        overlay = (0.4 * attn_array + 0.6 * image_array).astype(np.uint8)
        
        return Image.fromarray(overlay)
    
    def create_multi_head_visualization(
        self,
        image: Image.Image,
        attention_weights: np.ndarray,
        num_heads: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """
        Create a grid visualization of multiple attention heads.
        
        Args:
            image: Original PIL Image.
            attention_weights: Attention weights.
            num_heads: Number of heads to visualize.
            save_path: Optional path to save the visualization.
        
        Returns:
            Combined visualization image, or None on failure.
        """
        import matplotlib.pyplot as plt
        
        n_heads = num_heads or self.num_heads
        
        if attention_weights.ndim >= 3:
            total_heads = attention_weights.shape[-3] if attention_weights.ndim == 4 else attention_weights.shape[0]
            n_heads = min(n_heads, total_heads)
        else:
            n_heads = 1
        
        fig, axes = plt.subplots(1, n_heads + 1, figsize=(4 * (n_heads + 1), 4))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        # Attention head visualizations
        for i in range(n_heads):
            overlay = self.visualize_attention(image, attention_weights, head_idx=i)
            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(f"Head {i}")
            axes[i + 1].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Attention visualization saved to: {save_path}")
        
        # Convert plot to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        result = Image.fromarray(img_array)
        
        plt.close(fig)
        return result
    
    def cleanup_hooks(self):
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        self.cleanup_hooks()
