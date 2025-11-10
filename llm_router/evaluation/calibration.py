"""
Calibration Evaluation for LLM Router
Uses RMS calibration error from HLE (https://github.com/centerforaisafety/hle)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def calib_err(confidence: np.ndarray, correct: np.ndarray, p: str = '2', beta: int = 100) -> float:
    """
    Calculate calibration error using HLE's implementation.

    Args:
        confidence: Array of confidence scores (0.0 to 1.0)
        correct: Array of binary correctness (0 or 1)
        p: Norm type - '1' (L1), '2' (RMS/L2), 'infty' (max)
        beta: Target bin size (default 100)

    Returns:
        Calibration error (RMS by default)

    Reference:
        https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py#L97
    """
    # Validate inputs
    assert len(confidence) == len(correct), "confidence and correct must have same length"
    assert np.all((confidence >= 0) & (confidence <= 1)), "confidence must be in [0, 1]"
    assert np.all((correct == 0) | (correct == 1)), "correct must be binary (0 or 1)"

    # Sort by confidence
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]

    # Create bins
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)

    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def evaluate_router_calibration(
    predictions: List[Dict],
    beta: int = 100
) -> Dict[str, float]:
    """
    Evaluate calibration of router predictions.

    Args:
        predictions: List of dicts with keys:
            - 'confidence': float in [0, 1]
            - 'correct': bool or int (0/1)
        beta: Bin size for calibration error

    Returns:
        Dict with calibration metrics:
            - 'rms_calibration_error': RMS calibration error (L2)
            - 'l1_calibration_error': L1 calibration error
            - 'max_calibration_error': Max calibration error
            - 'accuracy': Overall accuracy
            - 'avg_confidence': Average confidence
            - 'num_examples': Number of examples
    """
    # Extract arrays
    confidence = np.array([p['confidence'] for p in predictions])
    correct = np.array([int(p['correct']) for p in predictions])

    # Validate confidence is in [0, 1]
    assert np.all((confidence >= 0) & (confidence <= 1)), \
        f"Confidence must be in [0, 1]. Got range [{confidence.min()}, {confidence.max()}]"

    # Calculate calibration errors
    rms_ce = calib_err(confidence, correct, p='2', beta=beta)
    l1_ce = calib_err(confidence, correct, p='1', beta=beta)
    max_ce = calib_err(confidence, correct, p='infty', beta=beta)

    # Additional metrics
    accuracy = np.mean(correct)
    avg_confidence = np.mean(confidence)

    return {
        'rms_calibration_error': float(rms_ce),
        'l1_calibration_error': float(l1_ce),
        'max_calibration_error': float(max_ce),
        'accuracy': float(accuracy),
        'avg_confidence': float(avg_confidence),
        'num_examples': len(predictions),
    }


def evaluate_by_model_size(
    predictions_by_size: Dict[str, List[Dict]],
    beta: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate calibration across different model sizes.

    Args:
        predictions_by_size: Dict mapping model size to predictions
            e.g., {'gemma-2b': [...], 'gemma-7b': [...]}
        beta: Bin size for calibration error

    Returns:
        Dict mapping model size to calibration metrics
    """
    results = {}

    for model_size, predictions in predictions_by_size.items():
        results[model_size] = evaluate_router_calibration(predictions, beta=beta)

    return results


def compute_oracle_confidence(
    medium_label: str,
    large_label: str,
    policy: str
) -> float:
    """
    Compute ground-truth confidence from oracle labels.

    This represents P(correct answer | routing policy decision)

    Args:
        medium_label: Judgment for medium model (Correct/Incorrect/IDK)
        large_label: Judgment for large model (Correct/Incorrect/IDK)
        policy: Chosen policy (Standard_Query/Complex_Query/Ambiguous_Query)

    Returns:
        Confidence score in [0, 1]
    """
    if policy == "Standard_Query":
        # Route to medium model - confident if medium is correct
        return 1.0 if medium_label == "Correct" else 0.0

    elif policy == "Complex_Query":
        # Route to large model - confident if large is correct
        return 1.0 if large_label == "Correct" else 0.0

    elif policy == "Ambiguous_Query":
        # Both failed - low confidence (route to cheap model to save cost)
        return 0.1  # Small non-zero for numerical stability

    else:
        return 0.5  # Default uncertainty


if __name__ == "__main__":
    # Example usage
    example_predictions = [
        {'confidence': 0.9, 'correct': True},
        {'confidence': 0.8, 'correct': True},
        {'confidence': 0.7, 'correct': False},
        {'confidence': 0.6, 'correct': True},
        {'confidence': 0.5, 'correct': False},
    ]

    metrics = evaluate_router_calibration(example_predictions, beta=2)
    print("Example Calibration Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
