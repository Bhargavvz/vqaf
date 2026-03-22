"""
Evaluation Metrics Module
===========================
Comprehensive metrics for Medical VQA evaluation including
accuracy, BLEU, ROUGE, and clinical consistency checks.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def normalize_answer(answer: str) -> str:
    """Normalize answer for evaluation."""
    if not answer:
        return ""
    answer = answer.lower().strip()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


def exact_match_accuracy(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute exact match accuracy after normalization.
    
    Args:
        predictions: List of predicted answers.
        references: List of ground-truth answers.
    
    Returns:
        Accuracy score between 0 and 1.
    """
    assert len(predictions) == len(references), "Length mismatch"
    
    correct = 0
    for pred, ref in zip(predictions, references):
        if normalize_answer(pred) == normalize_answer(ref):
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def fuzzy_match_accuracy(
    predictions: List[str],
    references: List[str],
    threshold: float = 0.85,
) -> float:
    """
    Compute fuzzy match accuracy using token-level F1.
    
    A prediction is considered correct if its token F1 score
    with the reference exceeds the threshold.
    
    Args:
        predictions: Predicted answers.
        references: Ground-truth answers.
        threshold: Minimum F1 score for a match.
    
    Returns:
        Fuzzy accuracy score.
    """
    correct = 0
    for pred, ref in zip(predictions, references):
        f1 = _token_f1(normalize_answer(pred), normalize_answer(ref))
        if f1 >= threshold:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between two strings."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    
    return 2 * precision * recall / (precision + recall)


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU scores for explanation quality.
    
    Args:
        predictions: Generated explanations.
        references: Reference explanations.
        max_n: Maximum n-gram order.
    
    Returns:
        Dictionary with BLEU-1 through BLEU-N scores.
    """
    try:
        from nltk.translate.bleu_score import (
            sentence_bleu,
            SmoothingFunction,
        )
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except ImportError:
        logger.warning("NLTK not available. Returning zero BLEU scores.")
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}
    
    smoothing = SmoothingFunction().method1
    bleu_scores = {f"bleu_{i}": [] for i in range(1, max_n + 1)}
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens or not ref_tokens:
            for i in range(1, max_n + 1):
                bleu_scores[f"bleu_{i}"].append(0.0)
            continue
        
        for i in range(1, max_n + 1):
            weights = tuple([1.0 / i] * i + [0.0] * (max_n - i))
            try:
                score = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=weights[:max_n],
                    smoothing_function=smoothing
                )
                bleu_scores[f"bleu_{i}"].append(score)
            except Exception:
                bleu_scores[f"bleu_{i}"].append(0.0)
    
    return {k: np.mean(v) for k, v in bleu_scores.items()}


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE scores for explanation quality.
    
    Args:
        predictions: Generated explanations.
        references: Reference explanations.
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score not available. Returning zeros.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            for key in scores:
                scores[key].append(0.0)
            continue
        
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)
    
    return {k: np.mean(v) for k, v in scores.items()}


# ============================================================
# Clinical Consistency Checks
# ============================================================

# Medical consistency rules
CLINICAL_RULES = {
    # If answer contains X, the explanation should mention related terms
    "cardiomegaly": ["heart", "cardiac", "cardiothoracic", "enlarged"],
    "pneumonia": ["lung", "consolidation", "infection", "infiltrate", "opacity"],
    "pleural effusion": ["pleural", "fluid", "costophrenic", "effusion"],
    "pneumothorax": ["lung", "air", "pleural", "collapse"],
    "fracture": ["bone", "break", "displaced", "cortex"],
    "normal": ["unremarkable", "within normal limits", "no abnormality", "normal"],
    "edema": ["fluid", "swelling", "pulmonary", "congestion"],
    "mass": ["lesion", "nodule", "tumor", "growth"],
    "atelectasis": ["collapse", "lung", "volume loss"],
}


def clinical_consistency_check(
    answers: List[str],
    explanations: List[str],
) -> Dict[str, Any]:
    """
    Check clinical consistency between answers and explanations.
    
    Verifies that when a specific condition is mentioned in the answer,
    the explanation contains related medical terms.
    
    Args:
        answers: Predicted answers.
        explanations: Generated explanations.
    
    Returns:
        Dictionary with consistency_score and detailed results.
    """
    total_checks = 0
    consistent = 0
    inconsistencies = []
    
    for i, (answer, explanation) in enumerate(zip(answers, explanations)):
        answer_lower = answer.lower()
        explanation_lower = explanation.lower()
        
        for condition, related_terms in CLINICAL_RULES.items():
            if condition in answer_lower:
                total_checks += 1
                
                # Check if explanation mentions any related term
                has_related = any(
                    term in explanation_lower
                    for term in related_terms
                )
                
                if has_related:
                    consistent += 1
                else:
                    inconsistencies.append({
                        "index": i,
                        "answer": answer,
                        "condition": condition,
                        "expected_terms": related_terms,
                    })
    
    consistency_score = consistent / total_checks if total_checks > 0 else 1.0
    
    return {
        "consistency_score": consistency_score,
        "total_checks": total_checks,
        "consistent": consistent,
        "inconsistencies": inconsistencies[:10],  # Limit output
    }


def compute_confusion_matrix(
    predictions: List[str],
    references: List[str],
    top_n_classes: int = 20,
) -> Dict[str, Any]:
    """
    Compute a confusion matrix for the top-N most common answer classes.
    
    Args:
        predictions: Predicted answers.
        references: Ground-truth answers.
        top_n_classes: Number of top classes to include.
    
    Returns:
        Dictionary with confusion matrix data and class labels.
    """
    # Normalize
    pred_norm = [normalize_answer(p) for p in predictions]
    ref_norm = [normalize_answer(r) for r in references]
    
    # Find top classes
    ref_counts = Counter(ref_norm)
    top_classes = [cls for cls, _ in ref_counts.most_common(top_n_classes)]
    
    # Build confusion matrix
    class_to_idx = {cls: i for i, cls in enumerate(top_classes)}
    n = len(top_classes)
    matrix = np.zeros((n, n), dtype=int)
    
    for pred, ref in zip(pred_norm, ref_norm):
        if ref in class_to_idx and pred in class_to_idx:
            matrix[class_to_idx[ref]][class_to_idx[pred]] += 1
    
    return {
        "matrix": matrix.tolist(),
        "labels": top_classes,
        "shape": (n, n),
    }


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    explanations: Optional[List[str]] = None,
    reference_explanations: Optional[List[str]] = None,
    fuzzy_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted answers.
        references: Ground-truth answers.
        explanations: Generated explanations (optional).
        reference_explanations: Reference explanations (optional).
        fuzzy_threshold: Threshold for fuzzy matching.
    
    Returns:
        Comprehensive metrics dictionary.
    """
    metrics = {}
    
    # Accuracy
    metrics["exact_match_accuracy"] = exact_match_accuracy(predictions, references)
    metrics["fuzzy_match_accuracy"] = fuzzy_match_accuracy(
        predictions, references, fuzzy_threshold
    )
    
    # Confusion matrix
    metrics["confusion_matrix"] = compute_confusion_matrix(predictions, references)
    
    # Explanation metrics (if available)
    if explanations and reference_explanations:
        metrics["bleu"] = compute_bleu(explanations, reference_explanations)
        metrics["rouge"] = compute_rouge(explanations, reference_explanations)
        metrics["clinical_consistency"] = clinical_consistency_check(
            predictions, explanations
        )
    elif explanations:
        metrics["clinical_consistency"] = clinical_consistency_check(
            predictions, explanations
        )
    
    # Per-class accuracy
    class_metrics = _per_class_accuracy(predictions, references)
    metrics["per_class_accuracy"] = class_metrics
    
    return metrics


def _per_class_accuracy(
    predictions: List[str],
    references: List[str],
    top_n: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class accuracy for top classes."""
    pred_norm = [normalize_answer(p) for p in predictions]
    ref_norm = [normalize_answer(r) for r in references]
    
    ref_counts = Counter(ref_norm)
    top_classes = [cls for cls, _ in ref_counts.most_common(top_n)]
    
    class_metrics = {}
    for cls in top_classes:
        indices = [i for i, r in enumerate(ref_norm) if r == cls]
        correct = sum(1 for i in indices if pred_norm[i] == cls)
        class_metrics[cls] = {
            "accuracy": correct / len(indices) if indices else 0.0,
            "total": len(indices),
            "correct": correct,
        }
    
    return class_metrics
