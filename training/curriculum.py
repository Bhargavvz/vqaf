"""
Curriculum Learning Module
============================
Implements staged training with increasing question difficulty.
Stages: easy (yes/no) → medium (counting, location) → hard (descriptive, reasoning)
"""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, Subset
from loguru import logger


class CurriculumScheduler:
    """
    Curriculum learning scheduler that provides staged training data.
    
    Stages questions from easy (yes/no, binary) through medium
    (counting, color, location) to hard (descriptive, reasoning).
    Each stage includes all previous stages' data plus the new stage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Curriculum configuration with stages definition.
        """
        self.enabled = config.get("enabled", True)
        self.stages = config.get("stages", [
            {"name": "easy", "question_types": ["yes/no", "binary"], "epochs": 1},
            {"name": "medium", "question_types": ["counting", "color", "location"], "epochs": 2},
            {"name": "hard", "question_types": ["descriptive", "reasoning", "comparison"], "epochs": 2},
        ])
        
        # Map difficulty levels to stage indices
        self.difficulty_to_stage = {
            "easy": 0,
            "medium": 1,
            "hard": 2,
        }
        
        logger.info(
            f"CurriculumScheduler initialized with {len(self.stages)} stages: "
            + ", ".join(s['name'] for s in self.stages)
        )
    
    def get_stage_for_epoch(self, epoch: int) -> int:
        """
        Determine which curriculum stage to use for a given epoch.
        
        Args:
            epoch: Current training epoch (0-indexed).
        
        Returns:
            Stage index (0=easy, 1=medium, 2=hard).
        """
        cumulative = 0
        for i, stage in enumerate(self.stages):
            cumulative += stage.get("epochs", 1)
            if epoch < cumulative:
                return i
        return len(self.stages) - 1  # Last stage for remaining epochs
    
    def get_stage_dataset(
        self,
        dataset: Dataset,
        epoch: int
    ) -> Dataset:
        """
        Get the subset of the dataset appropriate for the current
        curriculum stage.
        
        Curriculum is cumulative: each stage includes all easier stages.
        
        Args:
            dataset: Full training dataset (must have `all_samples` attribute).
            epoch: Current training epoch.
        
        Returns:
            Filtered dataset for the current stage.
        """
        if not self.enabled:
            return dataset
        
        current_stage = self.get_stage_for_epoch(epoch)
        stage_name = self.stages[current_stage]["name"]
        
        # Determine which difficulty levels to include
        # Cumulative: include all stages up to and including current
        allowed_difficulties = set()
        for i in range(current_stage + 1):
            allowed_difficulties.add(self.stages[i]["name"])
        
        # Get indices of matching samples
        if hasattr(dataset, 'all_samples'):
            indices = [
                i for i, sample in enumerate(dataset.all_samples)
                if sample.get("difficulty", "hard") in allowed_difficulties
            ]
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'all_samples'):
            indices = [
                i for i, sample in enumerate(dataset.dataset.all_samples)
                if sample.get("difficulty", "hard") in allowed_difficulties
            ]
        else:
            logger.warning("Dataset does not have 'all_samples'. Using full dataset.")
            return dataset
        
        logger.info(
            f"Curriculum stage '{stage_name}' (epoch {epoch}): "
            f"{len(indices)}/{len(dataset)} samples "
            f"(difficulties: {allowed_difficulties})"
        )
        
        return Subset(dataset, indices)
    
    def get_total_epochs(self) -> int:
        """Get total number of epochs across all stages."""
        return sum(stage.get("epochs", 1) for stage in self.stages)
    
    def get_stage_info(self) -> List[Dict[str, Any]]:
        """Get information about all curriculum stages."""
        info = []
        cumulative_epoch = 0
        for stage in self.stages:
            epochs = stage.get("epochs", 1)
            info.append({
                "name": stage["name"],
                "epochs": epochs,
                "start_epoch": cumulative_epoch,
                "end_epoch": cumulative_epoch + epochs - 1,
                "question_types": stage.get("question_types", []),
            })
            cumulative_epoch += epochs
        return info
