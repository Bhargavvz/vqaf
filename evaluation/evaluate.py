"""
Evaluation Script
==================
Full evaluation pipeline for the trained Medical VQA model.

Usage:
    python -m medical_vqa.evaluation.evaluate \
        --config medical_vqa/config.yaml \
        --model-path outputs/final_model \
        --output-dir outputs/evaluation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def main(args):
    """Run full evaluation pipeline."""
    
    # --------------------------------------------------------
    # 1. Load Configuration
    # --------------------------------------------------------
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = args.output_dir or config.get("paths", {}).get("output_dir", "./outputs")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("MEDICAL VQA EVALUATION PIPELINE")
    logger.info("=" * 60)
    
    # --------------------------------------------------------
    # 2. Load Model
    # --------------------------------------------------------
    logger.info("Loading model...")
    
    from medical_vqa.model.model import MedicalVQAModel
    
    model_wrapper = MedicalVQAModel(config)
    model_wrapper.load_model(for_training=False)
    
    if args.model_path:
        model_wrapper.load_trained_adapter(args.model_path)
        logger.info(f"Loaded adapter from: {args.model_path}")
    
    # --------------------------------------------------------
    # 3. Load Knowledge Retriever
    # --------------------------------------------------------
    logger.info("Loading knowledge retriever...")
    
    from medical_vqa.knowledge.retriever import MedicalKnowledgeRetriever
    
    knowledge_config = config.get("knowledge", {})
    retriever = MedicalKnowledgeRetriever(
        embedding_model_name=knowledge_config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        index_dir=config.get("paths", {}).get("knowledge_index_dir"),
        top_k=knowledge_config.get("top_k", 3),
    )
    
    try:
        retriever.load_index()
    except Exception:
        retriever.build_index()
    
    # --------------------------------------------------------
    # 4. Load Test Dataset
    # --------------------------------------------------------
    logger.info("Loading test dataset...")
    
    from medical_vqa.data.dataset import CombinedMedicalVQADataset
    
    test_dataset = CombinedMedicalVQADataset(
        config=config.get("dataset", {}),
        split="test",
        oversample_rare=False,
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # --------------------------------------------------------
    # 5. Run Inference on Test Set
    # --------------------------------------------------------
    logger.info("Running inference...")
    
    predictions = []
    references = []
    explanations = []
    
    for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
        sample = test_dataset[i]
        
        # Retrieve knowledge
        knowledge = retriever.retrieve_and_format(sample["question"])
        
        # Generate answer
        try:
            result = model_wrapper.generate(
                image=sample["image"],
                question=sample["question"],
                knowledge=knowledge,
                max_new_tokens=256,
                temperature=0.1,
            )
            
            predictions.append(result["answer"])
            explanations.append(result.get("explanation", ""))
        except Exception as e:
            logger.error(f"Inference failed for sample {i}: {e}")
            predictions.append("")
            explanations.append("")
        
        references.append(sample["answer"])
    
    # --------------------------------------------------------
    # 6. Compute Metrics
    # --------------------------------------------------------
    logger.info("Computing metrics...")
    
    from medical_vqa.evaluation.metrics import compute_all_metrics
    
    eval_config = config.get("evaluation", {})
    metrics = compute_all_metrics(
        predictions=predictions,
        references=references,
        explanations=explanations,
        fuzzy_threshold=eval_config.get("fuzzy_match_threshold", 0.85),
    )
    
    # --------------------------------------------------------
    # 7. Report Results
    # --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
    logger.info(f"Fuzzy Match Accuracy: {metrics['fuzzy_match_accuracy']:.4f}")
    
    if "bleu" in metrics:
        for k, v in metrics["bleu"].items():
            logger.info(f"  {k}: {v:.4f}")
    
    if "rouge" in metrics:
        for k, v in metrics["rouge"].items():
            logger.info(f"  {k}: {v:.4f}")
    
    if "clinical_consistency" in metrics:
        cc = metrics["clinical_consistency"]
        logger.info(f"Clinical Consistency: {cc['consistency_score']:.4f}")
    
    if "per_class_accuracy" in metrics:
        logger.info("\nPer-Class Accuracy (Top 10):")
        for cls, info in metrics["per_class_accuracy"].items():
            logger.info(f"  {cls}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")
    
    # --------------------------------------------------------
    # 8. Save Results
    # --------------------------------------------------------
    results_file = os.path.join(eval_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_file}")
    
    # Save predictions
    predictions_file = os.path.join(eval_dir, "predictions.json")
    with open(predictions_file, 'w') as f:
        pred_data = [
            {"prediction": p, "reference": r, "explanation": e}
            for p, r, e in zip(predictions, references, explanations)
        ]
        json.dump(pred_data, f, indent=2)
    logger.info(f"Predictions saved to: {predictions_file}")
    
    # Generate confusion matrix visualization
    try:
        _save_confusion_matrix(metrics.get("confusion_matrix", {}), eval_dir)
    except Exception as e:
        logger.warning(f"Could not save confusion matrix visualization: {e}")
    
    # Error analysis
    _error_analysis(predictions, references, explanations, eval_dir)
    
    logger.info("Evaluation complete!")
    return metrics


def _save_confusion_matrix(cm_data: Dict, output_dir: str):
    """Save confusion matrix as an image."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    if not cm_data.get("matrix"):
        return
    
    matrix = np.array(cm_data["matrix"])
    labels = cm_data["labels"]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Medical VQA")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to: {path}")


def _error_analysis(
    predictions: list,
    references: list,
    explanations: list,
    output_dir: str,
):
    """Perform error analysis and save report."""
    from medical_vqa.evaluation.metrics import normalize_answer
    
    errors = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_norm = normalize_answer(pred)
        ref_norm = normalize_answer(ref)
        
        if pred_norm != ref_norm:
            errors.append({
                "index": i,
                "predicted": pred,
                "predicted_normalized": pred_norm,
                "reference": ref,
                "reference_normalized": ref_norm,
                "explanation": explanations[i] if i < len(explanations) else "",
            })
    
    # Categorize errors
    error_categories = {
        "yes_no_confusion": [],
        "partial_match": [],
        "completely_wrong": [],
        "empty_prediction": [],
    }
    
    for error in errors:
        pred = error["predicted_normalized"]
        ref = error["reference_normalized"]
        
        if not pred:
            error_categories["empty_prediction"].append(error)
        elif {pred, ref} <= {"yes", "no"}:
            error_categories["yes_no_confusion"].append(error)
        elif pred in ref or ref in pred:
            error_categories["partial_match"].append(error)
        else:
            error_categories["completely_wrong"].append(error)
    
    report = {
        "total_samples": len(predictions),
        "total_errors": len(errors),
        "error_rate": len(errors) / len(predictions) if predictions else 0,
        "error_breakdown": {
            k: len(v) for k, v in error_categories.items()
        },
        "sample_errors": {
            k: v[:5] for k, v in error_categories.items()  # Top 5 per category
        },
    }
    
    report_file = os.path.join(output_dir, "error_analysis.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Error analysis saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Medical VQA Model")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained adapter weights"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for evaluation outputs"
    )
    args = parser.parse_args()
    
    main(args)
