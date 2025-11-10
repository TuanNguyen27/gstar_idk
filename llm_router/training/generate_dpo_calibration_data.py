#!/usr/bin/env python3
"""
Generate DPO preference pairs for calibration improvement.
Uses judged SLM results to create training data where models learn to output calibrated confidence.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.calibration import calib_err


def calculate_target_confidence(is_correct: bool, strategy: str = "binary") -> float:
    """
    Calculate target confidence for a prediction.

    Args:
        is_correct: Whether the answer was correct
        strategy: Strategy for target confidence
            - "binary": 0.9 for correct, 0.1 for incorrect
            - "moderate": 0.8 for correct, 0.2 for incorrect (more conservative)

    Returns:
        Target confidence in [0, 1]
    """
    if strategy == "binary":
        return 0.9 if is_correct else 0.1
    elif strategy == "moderate":
        return 0.8 if is_correct else 0.2
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_dpo_preference_pairs(
    judged_file: Path,
    output_file: Path,
    min_miscalibration: float = 0.3,
    confidence_strategy: str = "binary",
    include_well_calibrated: bool = False,
) -> List[Dict]:
    """
    Create DPO preference pairs from judged SLM results.

    Args:
        judged_file: Path to judged results JSONL
        output_file: Path to save DPO pairs
        min_miscalibration: Minimum |confidence - target| to include example
        confidence_strategy: Strategy for target confidence ("binary" or "moderate")
        include_well_calibrated: Whether to include well-calibrated examples as positive samples

    Returns:
        List of DPO preference pairs
    """
    print(f"Loading judged results from {judged_file}")

    # Load judged results
    judged_examples = []
    with open(judged_file, 'r') as f:
        for line in f:
            judged_examples.append(json.loads(line))

    print(f"Loaded {len(judged_examples)} judged examples")

    # Generate preference pairs
    preference_pairs = []

    for example in judged_examples:
        question = example['problem']
        answer = example['model_answer']
        original_confidence = example['confidence']
        is_correct = example['is_correct']

        # Calculate target confidence
        target_confidence = calculate_target_confidence(is_correct, confidence_strategy)

        # Calculate individual calibration error
        individual_error = abs(original_confidence - float(is_correct))

        # Only include if poorly calibrated (or if we want well-calibrated examples)
        if individual_error < min_miscalibration and not include_well_calibrated:
            continue

        # Create prompt
        prompt = f"""Answer the following question concisely and provide your confidence (0.0 to 1.0).

Question: {question}

Format your response as:
Answer: [your answer]
Confidence: [0.0-1.0]"""

        # Create chosen (well-calibrated) response
        chosen_response = f"Answer: {answer}\nConfidence: {target_confidence:.2f}"

        # Create rejected (poorly-calibrated) response
        rejected_response = f"Answer: {answer}\nConfidence: {original_confidence:.2f}"

        # DPO preference pair
        pair = {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "metadata": {
                "question": question,
                "answer": answer,
                "is_correct": is_correct,
                "original_confidence": original_confidence,
                "target_confidence": target_confidence,
                "individual_error": individual_error,
                "topic": example.get('topic', 'unknown'),
                "answer_type": example.get('answer_type', 'unknown'),
            }
        }

        preference_pairs.append(pair)

    # Save preference pairs
    print(f"\nGenerated {len(preference_pairs)} DPO preference pairs")
    print(f"Saving to {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for pair in preference_pairs:
            f.write(json.dumps(pair) + '\n')

    # Statistics
    correct_examples = sum(1 for p in preference_pairs if p['metadata']['is_correct'])
    incorrect_examples = len(preference_pairs) - correct_examples

    original_confidences = [p['metadata']['original_confidence'] for p in preference_pairs]
    target_confidences = [p['metadata']['target_confidence'] for p in preference_pairs]
    individual_errors = [p['metadata']['individual_error'] for p in preference_pairs]

    print(f"\n{'='*80}")
    print("DPO DATA GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total pairs: {len(preference_pairs)}")
    print(f"Correct answers: {correct_examples}")
    print(f"Incorrect answers: {incorrect_examples}")
    print(f"\nOriginal confidence: {np.mean(original_confidences):.3f} ± {np.std(original_confidences):.3f}")
    print(f"Target confidence: {np.mean(target_confidences):.3f} ± {np.std(target_confidences):.3f}")
    print(f"Avg individual error: {np.mean(individual_errors):.3f}")
    print(f"\nSaved to: {output_file}")

    return preference_pairs


def calculate_baseline_calibration(judged_file: Path, beta: int = 100) -> Dict[str, float]:
    """
    Calculate baseline calibration metrics using HLE's RMS error.

    Args:
        judged_file: Path to judged results JSONL
        beta: Bin size for calibration error

    Returns:
        Dict with calibration metrics
    """
    print(f"\nCalculating baseline calibration for {judged_file.name}")

    # Load data
    confidences = []
    corrects = []

    with open(judged_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            confidences.append(example['confidence'])
            corrects.append(int(example['is_correct']))

    confidence_array = np.array(confidences)
    correct_array = np.array(corrects)

    # Calculate calibration error
    rms_error = calib_err(confidence_array, correct_array, p='2', beta=beta)
    l1_error = calib_err(confidence_array, correct_array, p='1', beta=beta)

    accuracy = np.mean(corrects)
    avg_confidence = np.mean(confidences)

    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Avg confidence: {avg_confidence:.3f}")
    print(f"  RMS calibration error: {rms_error:.3f}")
    print(f"  L1 calibration error: {l1_error:.3f}")

    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'rms_calibration_error': rms_error,
        'l1_calibration_error': l1_error,
        'num_examples': len(confidences),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs for calibration improvement"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input judged results JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output DPO preference pairs JSONL file"
    )
    parser.add_argument(
        "--min-miscalibration",
        type=float,
        default=0.3,
        help="Minimum calibration error to include example (default: 0.3)"
    )
    parser.add_argument(
        "--confidence-strategy",
        type=str,
        choices=["binary", "moderate"],
        default="binary",
        help="Strategy for target confidence (default: binary)"
    )
    parser.add_argument(
        "--include-well-calibrated",
        action="store_true",
        help="Include well-calibrated examples as positive samples"
    )
    parser.add_argument(
        "--beta",
        type=int,
        default=100,
        help="Bin size for HLE calibration error calculation (default: 100)"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    # Calculate baseline calibration
    baseline_metrics = calculate_baseline_calibration(input_file, beta=args.beta)

    # Generate DPO preference pairs
    preference_pairs = create_dpo_preference_pairs(
        judged_file=input_file,
        output_file=output_file,
        min_miscalibration=args.min_miscalibration,
        confidence_strategy=args.confidence_strategy,
        include_well_calibrated=args.include_well_calibrated,
    )

    print(f"\n{'='*80}")
    print("BASELINE CALIBRATION METRICS")
    print(f"{'='*80}")
    for key, value in baseline_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
