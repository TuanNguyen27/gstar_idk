#!/usr/bin/env python3
"""
Measure calibration of routing decisions across different model sizes.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def measure_calibration(
    predictions: List[dict],
    n_bins: int = 10
) -> dict:
    """
    Measure calibration metrics.
    
    predictions: [{"confidence": 0.8, "correct": True}, ...]
    """
    confidences = [p["confidence"] for p in predictions]
    correct = [1 if p["correct"] else 0 for p in predictions]
    
    # Expected Calibration Error (ECE)
    prob_true, prob_pred = calibration_curve(correct, confidences, n_bins=n_bins)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred))
    
    # Brier Score
    brier = np.mean((np.array(confidences) - np.array(correct)) ** 2)
    
    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }

if __name__ == "__main__":
    print("Calibration measurement script ready")
