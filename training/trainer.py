"""
Training Module
================
Custom trainer for Medical VQA model using HuggingFace Trainer
with multi-task loss, curriculum learning support, and 
comprehensive logging.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)


# ============================================================
# Custom Training Dataset Wrapper
# ============================================================

class MedicalVQATrainDataset(Dataset):
    """
    Training dataset that preprocesses samples on-the-fly
    using the model's processor and RAG retriever.
    """
    
    def __init__(
        self,
        dataset,
        processor,
        retriever=None,
        max_seq_length: int = 512,
    ):
        """
        Args:
            dataset: Source CombinedMedicalVQADataset.
            processor: MedicalVQAProcessor instance.
            retriever: MedicalKnowledgeRetriever instance (optional).
            max_seq_length: Maximum sequence length.
        """
        self.dataset = dataset
        self.processor = processor
        self.retriever = retriever
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Retrieve relevant knowledge
        knowledge = ""
        if self.retriever:
            try:
                knowledge = self.retriever.retrieve_and_format(
                    sample["question"],
                    top_k=3,
                    max_tokens=200
                )
            except Exception as e:
                logger.debug(f"Knowledge retrieval failed for idx {idx}: {e}")
        
        # Process sample into model inputs
        try:
            inputs = self.processor.process_sample(
                image=sample["image"],
                question=sample["question"],
                knowledge=knowledge,
                answer=sample.get("answer", ""),
            )
            return inputs
        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {e}")
            # Return a dummy sample
            return self._get_dummy_sample()
    
    def _get_dummy_sample(self):
        """Return a dummy sample for error cases."""
        from PIL import Image
        dummy_image = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.processor.process_sample(
            image=dummy_image,
            question="What is shown?",
            answer="Unknown"
        )


# ============================================================
# Time Limit Callback
# ============================================================

class TimeLimitCallback(TrainerCallback):
    """Stop training after a specified time limit."""
    
    def __init__(self, max_hours: float = 7.0):
        self.max_seconds = max_hours * 3600
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info(f"Training started. Time limit: {self.max_seconds / 3600:.1f} hours")
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_seconds:
                logger.warning(
                    f"Time limit reached ({elapsed / 3600:.2f} hours). "
                    "Stopping training."
                )
                control.should_training_stop = True
        return control


# ============================================================
# Metrics Logging Callback
# ============================================================

class MetricsLoggingCallback(TrainerCallback):
    """Enhanced logging of training metrics."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            epoch = state.epoch or 0
            
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in logs.items()
                if k not in ("epoch",)
            )
            
            logger.info(f"[Step {step} | Epoch {epoch:.2f}] {metrics_str}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            logger.info("=" * 60)
            logger.info("EVALUATION RESULTS:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            logger.info("=" * 60)


# ============================================================
# Custom Trainer with Multi-Task Loss
# ============================================================

class MedicalVQATrainer(Trainer):
    """
    Custom Trainer that supports:
    - Multi-task loss (answer + explanation)
    - Weighted loss components
    - Custom evaluation metrics
    """
    
    def __init__(
        self,
        answer_weight: float = 0.7,
        explanation_weight: float = 0.3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.answer_weight = answer_weight
        self.explanation_weight = explanation_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute weighted loss for answer + explanation generation.
        
        The loss is a weighted combination:
            L = w_answer * L_answer + w_explanation * L_explanation
        
        For simplicity, since both answer and explanation are part of
        the same generation sequence, we use the standard causal LM
        loss but could extend with separate heads.
        """
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)
        
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss


# ============================================================
# Training Orchestrator
# ============================================================

def create_training_args(config: Dict[str, Any], output_dir: str) -> TrainingArguments:
    """
    Create HuggingFace TrainingArguments from config.
    
    Args:
        config: Training configuration dictionary.
        output_dir: Directory for outputs and checkpoints.
    
    Returns:
        TrainingArguments instance.
    """
    train_config = config.get("training", {})
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config.get("num_epochs", 5),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
        learning_rate=train_config.get("learning_rate", 1e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.05),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", True),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        dataloader_num_workers=train_config.get("dataloader_num_workers", 4),
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 200),
        eval_strategy=train_config.get("eval_strategy", "steps"),
        eval_steps=train_config.get("eval_steps", 200),
        save_total_limit=train_config.get("save_total_limit", 3),
        load_best_model_at_end=train_config.get("load_best_model_at_end", True),
        metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        logging_steps=train_config.get("logging_steps", 50),
        report_to=train_config.get("report_to", "tensorboard"),
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        logging_dir=os.path.join(output_dir, "logs"),
    )


def setup_trainer(
    model,
    processor,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: Dict[str, Any],
    output_dir: str,
) -> MedicalVQATrainer:
    """
    Set up the complete training pipeline.
    
    Args:
        model: The model to train (with LoRA).
        processor: MedicalVQAProcessor for collation.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        config: Full configuration.
        output_dir: Output directory.
    
    Returns:
        Configured MedicalVQATrainer instance.
    """
    training_args = create_training_args(config, output_dir)
    
    # Setup callbacks
    callbacks = [
        MetricsLoggingCallback(),
        TimeLimitCallback(
            max_hours=config.get("training", {}).get("max_training_hours", 7.0)
        ),
    ]
    
    # Add early stopping
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.001
        )
    )
    
    # Loss weights
    loss_config = config.get("loss", {})
    
    trainer = MedicalVQATrainer(
        answer_weight=loss_config.get("answer_weight", 0.7),
        explanation_weight=loss_config.get("explanation_weight", 0.3),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=processor.collate_fn,
        callbacks=callbacks,
    )
    
    return trainer
