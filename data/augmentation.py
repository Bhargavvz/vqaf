"""
Medical Image Augmentation Module
==================================
Provides image augmentation pipeline for medical VQA training data.
Includes both geometric and photometric augmentations, plus question
paraphrasing for text diversity.
"""

import random
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from loguru import logger


class MedicalImageAugmenter:
    """
    Medical image augmentation pipeline.
    
    Applies conservative augmentations suitable for medical images:
    - Random rotation (small angles to preserve anatomy)
    - Brightness/contrast adjustments
    - Horizontal flip (for applicable modalities)
    - Gaussian noise
    
    Note: Augmentations are conservative to preserve diagnostic information.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Augmentation configuration with parameters.
        """
        self.rotation = config.get("random_rotation", 15)
        self.brightness_range = config.get("brightness_range", [0.8, 1.2])
        self.contrast_range = config.get("contrast_range", [0.8, 1.2])
        self.flip_prob = config.get("horizontal_flip_prob", 0.3)
        self.noise_std = config.get("gaussian_noise_std", 0.02)
        
        logger.debug(
            f"MedicalImageAugmenter initialized: rotation={self.rotation}°, "
            f"brightness={self.brightness_range}, contrast={self.contrast_range}"
        )
    
    def augment(self, image: Image.Image) -> Image.Image:
        """
        Apply random augmentations to a medical image.
        
        Each augmentation is applied with a probability, ensuring
        variability while preserving diagnostic quality.
        
        Args:
            image: Input PIL Image.
        
        Returns:
            Augmented PIL Image.
        """
        if not isinstance(image, Image.Image):
            return image
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation, self.rotation)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
        
        # Random brightness adjustment
        if random.random() < 0.4:
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        # Random contrast adjustment
        if random.random() < 0.4:
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random Gaussian noise
        if random.random() < 0.3:
            image = self._add_gaussian_noise(image)
        
        return image
    
    def _add_gaussian_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to the image."""
        img_array = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.normal(0, self.noise_std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))


# ============================================================
# Question Paraphrasing
# ============================================================

# Paraphrase templates for common medical question types
PARAPHRASE_TEMPLATES = {
    "is_there": [
        "Is there {condition} visible in this image?",
        "Can you identify {condition} in this image?",
        "Does this image show {condition}?",
        "Is {condition} present in this scan?",
    ],
    "what_is": [
        "What is the finding in this image?",
        "What abnormality is shown?",
        "Can you describe what is seen in this image?",
        "What does this image demonstrate?",
    ],
    "where_is": [
        "Where is the abnormality located?",
        "In which region is the finding?",
        "What is the location of the pathology?",
        "Which area shows the abnormality?",
    ],
    "how_many": [
        "How many {objects} are visible?",
        "What is the count of {objects}?",
        "Can you count the {objects} in this image?",
    ],
}


def paraphrase_question(question: str) -> str:
    """
    Generate a paraphrased version of a medical question.
    
    Uses template matching to identify question type and randomly
    selects an alternative phrasing.
    
    Args:
        question: Original question text.
    
    Returns:
        Paraphrased question (or original if no template matches).
    """
    question_lower = question.lower().strip()
    
    # Simple random decision to keep original
    if random.random() < 0.5:
        return question
    
    # Try to match and paraphrase
    if question_lower.startswith(("is there", "are there")):
        templates = PARAPHRASE_TEMPLATES["is_there"]
        # Extract the condition part
        for prefix in ["is there", "are there"]:
            if question_lower.startswith(prefix):
                condition = question[len(prefix):].strip().rstrip("?").strip()
                if condition:
                    template = random.choice(templates)
                    return template.format(condition=condition)
    
    elif question_lower.startswith(("what is", "what are")):
        return random.choice(PARAPHRASE_TEMPLATES["what_is"])
    
    elif question_lower.startswith(("where is", "where are")):
        return random.choice(PARAPHRASE_TEMPLATES["where_is"])
    
    return question
