"""
Main Training Script
=====================
Entry point for training the Medical VQA model.

Usage:
    python -m medical_vqa.training.train --config config.yaml
    python -m medical_vqa.training.train --config config.yaml --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def setup_logging(config: Dict[str, Any]):
    """Setup file logging."""
    log_config = config.get("logging", {})
    log_file = log_config.get("file", "./outputs/logs/training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(log_file, level=log_config.get("level", "INFO"), rotation="100 MB")


def main(args):
    """Main training pipeline."""
    start_time = time.time()
    
    # --------------------------------------------------------
    # 1. Load Configuration
    # --------------------------------------------------------
    config = load_config(args.config)
    setup_logging(config)
    
    logger.info("=" * 70)
    logger.info("MEDICAL VQA TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # --------------------------------------------------------
    # 2. Setup Output Directories
    # --------------------------------------------------------
    output_dir = config.get("paths", {}).get("output_dir", "./outputs")
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", f"{output_dir}/checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # --------------------------------------------------------
    # 3. Build Knowledge Index (RAG)
    # --------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Building Knowledge Index (RAG)...")
    logger.info("-" * 40)
    
    from medical_vqa.knowledge.retriever import MedicalKnowledgeRetriever
    
    knowledge_config = config.get("knowledge", {})
    retriever = MedicalKnowledgeRetriever(
        embedding_model_name=knowledge_config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        index_dir=config.get("paths", {}).get("knowledge_index_dir", f"{output_dir}/knowledge_index"),
        top_k=knowledge_config.get("top_k", 3),
    )
    
    if knowledge_config.get("rebuild_index", False):
        retriever.build_index()
    else:
        try:
            retriever.load_index()
        except Exception:
            logger.info("No existing index found. Building new index...")
            retriever.build_index()
    
    logger.info(f"Knowledge index ready: {retriever.index.ntotal} entries")
    
    # --------------------------------------------------------
    # 4. Load Model
    # --------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Loading Model with QLoRA...")
    logger.info("-" * 40)
    
    from medical_vqa.model.model import MedicalVQAModel
    from medical_vqa.model.processor import MedicalVQAProcessor
    
    model_wrapper = MedicalVQAModel(config)
    model_wrapper.load_model(for_training=True)
    
    processor = MedicalVQAProcessor(
        processor=model_wrapper.processor,
        max_seq_length=config.get("dataset", {}).get("max_seq_length", 512),
    )
    
    model_info = model_wrapper.get_model_info()
    for k, v in model_info.items():
        logger.info(f"  {k}: {v}")
    
    if args.dry_run:
        logger.info("DRY RUN: Model loaded successfully. Exiting.")
        return
    
    # --------------------------------------------------------
    # 5. Load Datasets
    # --------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Loading Datasets...")
    logger.info("-" * 40)
    
    from medical_vqa.data.dataset import CombinedMedicalVQADataset
    from medical_vqa.training.trainer import MedicalVQATrainDataset
    
    dataset_config = config.get("dataset", {})
    
    # Create raw datasets
    train_raw = CombinedMedicalVQADataset(
        config=dataset_config, split="train", oversample_rare=True
    )
    val_raw = CombinedMedicalVQADataset(
        config=dataset_config, split="val", oversample_rare=False
    )
    
    # Wrap with processor and retriever
    train_dataset = MedicalVQATrainDataset(
        dataset=train_raw,
        processor=processor,
        retriever=retriever,
        max_seq_length=dataset_config.get("max_seq_length", 512),
    )
    eval_dataset = MedicalVQATrainDataset(
        dataset=val_raw,
        processor=processor,
        retriever=retriever,
        max_seq_length=dataset_config.get("max_seq_length", 512),
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # --------------------------------------------------------
    # 6. Setup Curriculum Learning
    # --------------------------------------------------------
    curriculum_config = config.get("curriculum", {})
    if curriculum_config.get("enabled", False):
        from medical_vqa.training.curriculum import CurriculumScheduler
        curriculum = CurriculumScheduler(curriculum_config)
        logger.info(f"Curriculum learning enabled: {curriculum.get_total_epochs()} total epochs")
        for stage_info in curriculum.get_stage_info():
            logger.info(f"  Stage '{stage_info['name']}': epochs {stage_info['start_epoch']}-{stage_info['end_epoch']}")
    
    # --------------------------------------------------------
    # 7. Setup Trainer
    # --------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Setting up Trainer...")
    logger.info("-" * 40)
    
    from medical_vqa.training.trainer import setup_trainer
    
    trainer = setup_trainer(
        model=model_wrapper.model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        output_dir=checkpoint_dir,
    )
    
    # --------------------------------------------------------
    # 8. Train!
    # --------------------------------------------------------
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    
    # Resume from checkpoint if available
    resume_from = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = [
            d for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            resume_from = os.path.join(checkpoint_dir, latest)
            logger.info(f"Resuming from checkpoint: {resume_from}")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    
    # --------------------------------------------------------
    # 9. Save Final Model
    # --------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Saving final model...")
    logger.info("-" * 40)
    
    final_model_dir = os.path.join(output_dir, "final_model")
    model_wrapper.save_adapter(final_model_dir)
    
    # Save processor/tokenizer
    model_wrapper.processor.save_pretrained(final_model_dir)
    
    # --------------------------------------------------------
    # 10. Training Summary
    # --------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed / 3600:.2f} hours")
    logger.info(f"Training metrics: {train_result.metrics}")
    logger.info(f"Model saved to: {final_model_dir}")
    
    # Run final evaluation
    logger.info("-" * 40)
    logger.info("Running final evaluation...")
    logger.info("-" * 40)
    
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Save training metrics
    import json
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_results,
            "training_time_hours": elapsed / 3600,
            "model_info": model_info,
        }, f, indent=2, default=str)
    
    logger.info(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical VQA Model")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only load model and verify setup, don't train"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    
    main(args)
