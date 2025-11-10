#!/usr/bin/env python3
"""
Generate standard SFT data for calibration training.
Simpler than DPO - just direct supervision on (question, answer+confidence) pairs.
"""

import json
from pathlib import Path


def generate_sft_calibration_data(
    judged_file: Path,
    output_file: Path,
    include_ground_truth: bool = True,
):
    """
    Generate SFT training data that teaches both calibration AND accuracy.

    Strategy:
    - Incorrect answers: Use ground truth with high confidence (0.9)
    - Correct answers: Use model's answer with high confidence (0.9)

    Args:
        judged_file: Path to judged results with correctness labels
        output_file: Where to save SFT data
        include_ground_truth: If True, use ground truth for incorrect answers
    """
    print(f"Loading judged data from {judged_file}")
    judged_examples = []
    with open(judged_file, 'r') as f:
        for line in f:
            judged_examples.append(json.loads(line))

    print(f"Loaded {len(judged_examples)} examples")

    # Generate SFT examples
    sft_examples = []
    correct_count = 0
    incorrect_count = 0

    for ex in judged_examples:
        question = ex["problem"]
        model_answer = ex["model_answer"]
        ground_truth = ex.get("expected_answer", "")
        is_correct = ex["is_correct"]

        # Input prompt
        input_text = (
            "Answer the following question concisely and provide your confidence (0.0 to 1.0).\\n\\n"
            f"Question: {question}\\n\\n"
            "Format your response as:\\n"
            "Answer: [your answer]\\n"
            "Confidence: [0.0-1.0]"
        )

        if is_correct:
            # CORRECT: Output model's answer with high confidence
            correct_count += 1
            output_text = f"Answer: {model_answer}\\nConfidence: 0.90"
        else:
            # INCORRECT: Output ground truth (if enabled) or low confidence
            incorrect_count += 1
            if include_ground_truth:
                output_text = f"Answer: {ground_truth}\\nConfidence: 0.90"
            else:
                output_text = f"Answer: {model_answer}\\nConfidence: 0.10"

        sft_example = {
            "input": input_text,
            "output": output_text,
            "metadata": {
                "question": question,
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
            }
        }

        sft_examples.append(sft_example)

    # Save SFT data
    print(f"\\nGenerated {len(sft_examples)} SFT examples:")
    print(f"  - Correct answers: {correct_count}")
    print(f"  - Incorrect answers: {incorrect_count}")

    with open(output_file, 'w') as f:
        for example in sft_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\\nSaved to {output_file}")

    # Show examples
    print("\\n" + "="*80)
    print("Example SFT pairs:")
    print("="*80)

    # Show 1 correct example
    correct_ex = next((ex for ex in sft_examples if ex["metadata"]["is_correct"]), None)
    if correct_ex:
        print("\\n1. CORRECT answer:")
        print(f"   Question: {correct_ex['metadata']['question'][:60]}...")
        print(f"   Input: {correct_ex['input'][:100]}...")
        print(f"   Output: {correct_ex['output']}")

    # Show 1 incorrect example
    incorrect_ex = next((ex for ex in sft_examples if not ex["metadata"]["is_correct"]), None)
    if incorrect_ex:
        print("\\n2. INCORRECT answer (with ground truth):")
        print(f"   Question: {incorrect_ex['metadata']['question'][:60]}...")
        print(f"   Model answer: {incorrect_ex['metadata']['model_answer']}")
        print(f"   Ground truth: {incorrect_ex['metadata']['ground_truth']}")
        print(f"   Output: {incorrect_ex['output']}")

    print("\\n" + "="*80)

    return sft_examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT calibration data")
    parser.add_argument(
        "--judged-data",
        type=Path,
        default=Path("data/slm_baseline/google_gemma-2-2b-it_judged.jsonl"),
        help="Path to judged results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/slm_baseline/google_gemma-2-2b-it_sft_calibration.jsonl"),
        help="Output path for SFT data"
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Don't use ground truth for incorrect answers (just teach low confidence)"
    )

    args = parser.parse_args()

    generate_sft_calibration_data(
        judged_file=args.judged_data,
        output_file=args.output,
        include_ground_truth=not args.no_ground_truth,
    )
