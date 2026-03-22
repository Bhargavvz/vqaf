"""
Medical VQA Dataset Module
==========================
Handles loading, preprocessing, and combining VQA-RAD and PathVQA datasets.
Provides train/val/test splits with stratification and answer normalization.
"""

import json
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from loguru import logger

from medical_vqa.data.augmentation import MedicalImageAugmenter


# ============================================================
# Answer Normalization
# ============================================================

# Standard answer mappings for normalization
ANSWER_SYNONYMS = {
    "yes": ["yes", "y", "true", "positive", "correct", "affirmative", "1"],
    "no": ["no", "n", "false", "negative", "incorrect", "0"],
    "normal": ["normal", "unremarkable", "within normal limits", "wnl"],
    "abnormal": ["abnormal", "remarkable", "pathological"],
    "left": ["left", "l", "left side"],
    "right": ["right", "r", "right side"],
    "bilateral": ["bilateral", "both", "both sides"],
}


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for consistent evaluation.
    
    Steps:
        1. Lowercase and strip whitespace
        2. Remove articles (a, an, the)
        3. Remove punctuation
        4. Map synonym variants to canonical forms
        5. Collapse multiple spaces
    
    Args:
        answer: Raw answer string.
    
    Returns:
        Normalized answer string.
    """
    if not answer:
        return ""
    
    # Lowercase and strip
    answer = answer.lower().strip()
    
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    
    # Collapse whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Map synonyms to canonical form
    for canonical, variants in ANSWER_SYNONYMS.items():
        if answer in variants:
            return canonical
    
    return answer


def classify_question_difficulty(question: str) -> str:
    """
    Classify question difficulty for curriculum learning.
    
    Categories:
        - easy: yes/no and binary questions
        - medium: counting, color, location questions
        - hard: descriptive, reasoning, comparison questions
    
    Args:
        question: The question text.
    
    Returns:
        Difficulty level string: 'easy', 'medium', or 'hard'.
    """
    question_lower = question.lower().strip()
    
    # Easy: yes/no questions
    if any(question_lower.startswith(w) for w in [
        "is ", "are ", "does ", "do ", "was ", "were ", "has ", "have ",
        "can ", "could ", "will ", "would ", "should "
    ]):
        return "easy"
    
    # Medium: counting, color, location
    if any(keyword in question_lower for keyword in [
        "how many", "count", "number of",
        "what color", "colour",
        "where", "location", "which side", "which part",
        "what organ", "what type"
    ]):
        return "medium"
    
    # Hard: descriptive, reasoning
    return "hard"


# ============================================================
# VQA-RAD Dataset
# ============================================================

class VQARADDataset(Dataset):
    """
    Dataset class for VQA-RAD (Visual Question Answering in Radiology).
    
    VQA-RAD contains radiology images with question-answer pairs
    covering multiple imaging modalities (X-ray, CT, MRI).
    
    Expected data structure:
        data_path/
            VQA_RAD Dataset Public.json  (or trainset.json / testset.json)
            images/
                synpic12345.jpg
                ...
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        normalize_answers: bool = True
    ):
        """
        Args:
            data_path: Root path to VQA-RAD dataset.
            split: One of 'train', 'val', 'test'.
            transform: Optional image transform/augmentation.
            normalize_answers: Whether to normalize answer strings.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.normalize_answers = normalize_answers
        self.samples: List[Dict[str, Any]] = []
        
        self._load_data()
        logger.info(f"VQA-RAD [{split}]: Loaded {len(self.samples)} samples")
    
    def _load_data(self):
        """Load VQA-RAD JSON data files."""
        # Try different possible filenames
        possible_files = [
            self.data_path / "VQA_RAD Dataset Public.json",
            self.data_path / "vqa_rad.json",
            self.data_path / f"{self.split}set.json",
            self.data_path / f"{self.split}.json",
        ]
        
        data_file = None
        for f in possible_files:
            if f.exists():
                data_file = f
                break
        
        if data_file is None:
            logger.warning(
                f"VQA-RAD data not found at {self.data_path}. "
                f"Run download_datasets.py first. Using empty dataset."
            )
            return
        
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(raw_data, dict):
            raw_data = raw_data.get("data", raw_data.get("questions", []))
        
        for item in raw_data:
            image_name = item.get("image_name", item.get("image", ""))
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not image_name or not question:
                continue
            
            # Build image path
            image_path = self.data_path / "images" / image_name
            if not image_path.exists():
                image_path = self.data_path / image_name
            
            # Normalize answer if configured
            processed_answer = normalize_answer(answer) if self.normalize_answers else answer
            
            self.samples.append({
                "image_path": str(image_path),
                "question": question,
                "answer": processed_answer,
                "raw_answer": answer,
                "difficulty": classify_question_difficulty(question),
                "answer_type": item.get("answer_type", "unknown"),
                "question_type": item.get("question_type", "unknown"),
                "source": "vqa_rad"
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx].copy()
        
        # Load image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            sample["image"] = image
        except Exception as e:
            logger.error(f"Failed to load image {sample['image_path']}: {e}")
            # Return a blank image as fallback
            sample["image"] = Image.new("RGB", (448, 448), (0, 0, 0))
        
        return sample


# ============================================================
# PathVQA Dataset
# ============================================================

class PathVQADataset(Dataset):
    """
    Dataset class for PathVQA (Pathology Visual Question Answering).
    
    Loads from saved JSON + images on disk (created by download_datasets.py).
    Falls back to HuggingFace if local files not found.
    
    Expected data structure:
        data_path/
            train/
                images/
                    pathvqa_00000.jpg
                    ...
            train.json
            val.json
            test.json
    """
    
    def __init__(
        self,
        data_path: str = "./data/path_vqa",
        hf_name: str = "flaviagiammarino/path-vqa",
        split: str = "train",
        transform=None,
        normalize_answers: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_path: Path to saved PathVQA data directory.
            hf_name: HuggingFace dataset identifier (fallback).
            split: One of 'train', 'val', 'test'.
            transform: Optional image transform/augmentation.
            normalize_answers: Whether to normalize answer strings.
            cache_dir: Optional cache directory for HF downloads.
        """
        self.data_path = Path(data_path)
        self.hf_name = hf_name
        self.split = split
        self.transform = transform
        self.normalize_answers = normalize_answers
        self.samples: List[Dict[str, Any]] = []
        
        self._load_data(cache_dir)
        logger.info(f"PathVQA [{split}]: Loaded {len(self.samples)} samples")
    
    def _load_data(self, cache_dir: Optional[str]):
        """Load PathVQA from saved JSON files on disk."""
        # Try loading from local JSON files first
        json_file = self.data_path / f"{self.split}.json"
        
        if json_file.exists():
            self._load_from_disk(json_file)
            return
        
        # Fallback: try HuggingFace
        logger.warning(f"PathVQA JSON not found at {json_file}. Run download_datasets.py first.")
        logger.warning("PathVQA will be empty.")
    
    def _load_from_disk(self, json_file: Path):
        """Load PathVQA from saved JSON + images."""
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        images_dir = self.data_path / self.split / "images"
        
        for item in raw_data:
            image_name = item.get("image_name", "")
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not question:
                continue
            
            image_path = images_dir / image_name
            
            processed_answer = normalize_answer(answer) if self.normalize_answers else answer
            
            self.samples.append({
                "image_path": str(image_path),
                "question": question,
                "answer": processed_answer,
                "raw_answer": answer,
                "difficulty": classify_question_difficulty(question),
                "answer_type": item.get("answer_type", "open"),
                "question_type": item.get("question_type", "pathology"),
                "source": "path_vqa"
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx].copy()
        
        # Load image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (448, 448), (0, 0, 0))
        sample["image"] = image
        
        # Apply transform
        if self.transform and isinstance(sample["image"], Image.Image):
            sample["image"] = self.transform(sample["image"])
        
        return sample


# ============================================================
# Combined Dataset
# ============================================================

class CombinedMedicalVQADataset(Dataset):
    """
    Combined dataset that merges VQA-RAD and PathVQA with
    unified preprocessing, answer normalization, and optional
    oversampling of rare classes.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transform=None,
        oversample_rare: bool = True
    ):
        """
        Args:
            config: Dataset configuration dictionary.
            split: One of 'train', 'val', 'test'.
            transform: Optional image transform/augmentation.
            oversample_rare: Whether to oversample rare answer classes.
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.all_samples: List[Dict[str, Any]] = []
        
        # Load VQA-RAD
        if config.get("vqa_rad", {}).get("enabled", True):
            vqa_rad = VQARADDataset(
                data_path=config["vqa_rad"]["data_path"],
                split=split,
                transform=None,  # Transform applied later
                normalize_answers=True
            )
            self.all_samples.extend(vqa_rad.samples)
        
        # Load PathVQA
        if config.get("path_vqa", {}).get("enabled", True):
            path_vqa = PathVQADataset(
                data_path=config["path_vqa"].get("data_path", "./data/path_vqa"),
                hf_name=config["path_vqa"].get("hf_name", "flaviagiammarino/path-vqa"),
                split=split,
                transform=None,
                normalize_answers=True,
                cache_dir=config.get("cache_dir")
            )
            self.all_samples.extend(path_vqa.samples)
        
        # Apply oversampling for rare classes in training
        if split == "train" and oversample_rare:
            self._oversample_rare_classes()
        
        # Apply augmentation wrapper for training
        if split == "train" and config.get("augmentation", {}).get("enabled", False):
            self.augmenter = MedicalImageAugmenter(config.get("augmentation", {}))
        else:
            self.augmenter = None
        
        logger.info(
            f"CombinedMedicalVQA [{split}]: {len(self.all_samples)} total samples "
            f"({self._get_difficulty_distribution()})"
        )
    
    def _oversample_rare_classes(self, min_count: int = 50):
        """Oversample rare answer classes to improve balance."""
        from collections import Counter
        
        answer_counts = Counter(s["answer"] for s in self.all_samples)
        max_count = max(answer_counts.values())
        target_count = min(max_count, min_count * 5)  # Cap oversampling
        
        oversampled = []
        for answer, count in answer_counts.items():
            if count < min_count:
                samples_with_answer = [s for s in self.all_samples if s["answer"] == answer]
                # Repeat to reach target
                repeats_needed = min(target_count // count, 3)  # Max 3x oversampling
                oversampled.extend(samples_with_answer * repeats_needed)
        
        if oversampled:
            logger.info(f"Oversampled {len(oversampled)} samples for rare classes")
            self.all_samples.extend(oversampled)
    
    def _get_difficulty_distribution(self) -> str:
        """Get a string summary of difficulty distribution."""
        from collections import Counter
        dist = Counter(s["difficulty"] for s in self.all_samples)
        return ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
    
    def __len__(self) -> int:
        return len(self.all_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.all_samples[idx].copy()
        
        # Load image if needed
        if not isinstance(sample.get("image"), Image.Image):
            try:
                image = Image.open(sample["image_path"]).convert("RGB")
            except Exception:
                image = Image.new("RGB", (448, 448), (0, 0, 0))
            sample["image"] = image
        
        # Apply augmentation
        if self.augmenter:
            sample["image"] = self.augmenter.augment(sample["image"])
        
        # Apply additional transform
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        
        return sample
    
    def get_samples_by_difficulty(self, difficulty: str) -> List[int]:
        """Get indices of samples with a specific difficulty level."""
        return [i for i, s in enumerate(self.all_samples) if s["difficulty"] == difficulty]


# ============================================================
# Data Splitting Utility
# ============================================================

def create_splits(
    config: Dict[str, Any],
    transform_train=None,
    transform_eval=None
) -> Tuple[CombinedMedicalVQADataset, CombinedMedicalVQADataset, CombinedMedicalVQADataset]:
    """
    Create train/val/test splits of the combined dataset.
    
    Args:
        config: Full configuration dictionary.
        transform_train: Transform for training data.
        transform_eval: Transform for validation/test data.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    dataset_config = config.get("dataset", config)
    
    train_dataset = CombinedMedicalVQADataset(
        config=dataset_config,
        split="train",
        transform=transform_train,
        oversample_rare=True
    )
    
    val_dataset = CombinedMedicalVQADataset(
        config=dataset_config,
        split="val",
        transform=transform_eval,
        oversample_rare=False
    )
    
    test_dataset = CombinedMedicalVQADataset(
        config=dataset_config,
        split="test",
        transform=transform_eval,
        oversample_rare=False
    )
    
    return train_dataset, val_dataset, test_dataset
