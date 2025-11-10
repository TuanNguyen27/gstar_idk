#!/usr/bin/env python3
"""
Generate improved DPO preference pairs with ground truth for incorrect answers.
Nudges confidence higher for correct answers instead of forcing to 0.9.
"""

import json
import random
from pathlib import Path


def generate_improved_dpo_pairs(
    judged_file: Path,
    output_file: Path,
    confidence_nudge: float = 0.2,
    nudge_variance: float = 0.1,
    max_confidence: float = 0.95,
    min_confidence: float = 0.05,
    random_seed: int = 42,
):
    """
    Generate DPO pairs that teach both calibration AND accuracy.

    Strategy:
    - Incorrect answers: prefer ground truth (0.9) over model answer with nudged-down confidence
    - Correct answers: nudge confidence higher by confidence_nudge
    - Nudge amount is randomized per example: nudge ± variance

    Args:
        judged_file: Path to judged results with correctness labels
        output_file: Where to save DPO pairs
        confidence_nudge: Base amount to increase/decrease confidence
        nudge_variance: Random variance around nudge (uniform ±variance)
        max_confidence: Maximum confidence to assign for correct answers
        min_confidence: Minimum confidence for incorrect answers
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    print(f"Loading judged data from {judged_file}")
    judged_examples = []
    with open(judged_file, 'r') as f:
        for line in f:
            judged_examples.append(json.loads(line))

    print(f"Loaded {len(judged_examples)} examples")

    # Generate DPO pairs
    dpo_pairs = []
    correct_count = 0
    incorrect_count = 0

    for ex in judged_examples:
        question = ex["problem"]
        model_answer = ex["model_answer"]
        ground_truth = ex.get("expected_answer", "")
        is_correct = ex["is_correct"]
        original_conf = ex.get("confidence", 0.5)

        # Base prompt
        prompt = (
            "Answer the following question concisely and provide your confidence (0.0 to 1.0).\n\n"
            f"Question: {question}\n\n"
            "Format your response as:\n"
            "Answer: [your answer]\n"
            "Confidence: [0.0-1.0]"
        )

        if is_correct:
            # CORRECT: Nudge confidence higher
            correct_count += 1

            # Randomize nudge amount: base ± variance
            actual_nudge = confidence_nudge + random.uniform(-nudge_variance, nudge_variance)

            # Calculate nudged confidence (but don't exceed max_confidence)
            nudged_conf = min(original_conf + actual_nudge, max_confidence)

            chosen = f"Answer: {model_answer}\nConfidence: {nudged_conf:.2f}"
            rejected = f"Answer: {model_answer}\nConfidence: {original_conf:.2f}"

        else:
            # INCORRECT: Prefer ground truth over model answer
            incorrect_count += 1

            # Chosen: Ground truth with high confidence
            chosen = f"Answer: {ground_truth}\nConfidence: 0.90"

            # Rejected: Model's wrong answer with nudged-down confidence
            # Randomize nudge amount: base ± variance
            actual_nudge = confidence_nudge + random.uniform(-nudge_variance, nudge_variance)

            # Nudge confidence DOWN (but don't go below min_confidence)
            nudged_down_conf = max(original_conf - actual_nudge, min_confidence)
            rejected = f"Answer: {model_answer}\nConfidence: {nudged_down_conf:.2f}"

        dpo_pair = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "question": question,
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "original_confidence": original_conf,
                "strategy": "nudge" if is_correct else "ground_truth",
            }
        }

        dpo_pairs.append(dpo_pair)

    # Save DPO pairs
    print(f"\nGenerated {len(dpo_pairs)} DPO pairs:")
    print(f"  - Correct answers (nudged): {correct_count}")
    print(f"  - Incorrect answers (ground truth): {incorrect_count}")

    with open(output_file, 'w') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"\nSaved to {output_file}")

    # Show examples
    print("\n" + "="*80)
    print("Example DPO pairs:")
    print("="*80)

    # Show 1 correct example
    correct_ex = next((p for p in dpo_pairs if p["metadata"]["is_correct"]), None)
    if correct_ex:
        print("\n1. CORRECT answer (nudge confidence):")
        print(f"   Question: {correct_ex['metadata']['question'][:60]}...")
        print(f"   Model answer: {correct_ex['metadata']['model_answer']}")
        print(f"   Original confidence: {correct_ex['metadata']['original_confidence']:.2f}")
        print(f"\n   Chosen: {correct_ex['chosen']}")
        print(f"   Rejected: {correct_ex['rejected']}")

    # Show 1 incorrect example
    incorrect_ex = next((p for p in dpo_pairs if not p["metadata"]["is_correct"]), None)
    if incorrect_ex:
        print("\n2. INCORRECT answer (prefer ground truth):")
        print(f"   Question: {incorrect_ex['metadata']['question'][:60]}...")
        print(f"   Model answer: {incorrect_ex['metadata']['model_answer']}")
        print(f"   Ground truth: {incorrect_ex['metadata']['ground_truth']}")
        print(f"   Original confidence: {incorrect_ex['metadata']['original_confidence']:.2f}")
        print(f"\n   Chosen: {incorrect_ex['chosen']}")
        print(f"   Rejected: {incorrect_ex['rejected']}")

    print("\n" + "="*80)

    return dpo_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate improved DPO pairs")
    parser.add_argument(
        "--judged-data",
        type=Path,
        default=Path("data/slm_baseline/google_gemma-2-2b-it_judged.jsonl"),
        help="Path to judged results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/slm_baseline/google_gemma-2-2b-it_dpo_pairs_v2.jsonl"),
        help="Output path for DPO pairs"
    )
    parser.add_argument(
        "--confidence-nudge",
        type=float,
        default=0.2,
        help="Base amount to increase/decrease confidence"
    )
    parser.add_argument(
        "--nudge-variance",
        type=float,
        default=0.1,
        help="Random variance around nudge amount (uniform ±variance)"
    )

    args = parser.parse_args()

    generate_improved_dpo_pairs(
        judged_file=args.judged_data,
        output_file=args.output,
        confidence_nudge=args.confidence_nudge,
        nudge_variance=args.nudge_variance,
    )
