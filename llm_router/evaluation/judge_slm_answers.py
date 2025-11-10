#!/usr/bin/env python3
"""
Judge SLM answers using Gemini Pro as frontier model judge.
Compares model answers against ground truth for correctness.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gemini_client import GeminiClient


def create_judge_prompt(question: str, ground_truth: str, model_answer: str) -> str:
    """
    Create prompt for Gemini judge to evaluate answer correctness.

    Args:
        question: The question asked
        ground_truth: The correct answer
        model_answer: The model's answer to evaluate

    Returns:
        Judge prompt
    """
    return f"""You are an expert judge evaluating whether a model's answer is correct.

Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

Task: Determine if the model's answer is semantically equivalent to the ground truth.
- Consider answers correct if they convey the same information, even with different wording
- For numerical answers, allow minor formatting differences (e.g., "120,000" vs "120000")
- Ignore extra explanation or context, focus on the core answer
- Mark incorrect if the model provides wrong information or fails to answer

Respond with ONLY "Correct" or "Incorrect", nothing else."""


def judge_answers(
    input_file: Path,
    output_file: Path,
    gemini_api_key: str,
    batch_size: int = 10,
) -> List[Dict]:
    """
    Judge SLM answers using Gemini Pro.

    Args:
        input_file: Path to SLM results JSONL (with model_answer, ground_truth)
        output_file: Path to save judged results
        gemini_api_key: Gemini API key
        batch_size: Batch size for API calls

    Returns:
        List of judged results with correctness labels
    """
    # Initialize Gemini client
    client = GeminiClient(api_key=gemini_api_key)

    # Load SLM results
    print(f"Loading results from {input_file}")
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(results)} examples")
    print(f"Judging with Gemini Pro (gemini-2.0-flash-exp)...")

    # Judge each answer
    judged_results = []

    for i, result in enumerate(tqdm(results, desc="Judging")):
        # Create judge prompt
        prompt = create_judge_prompt(
            question=result['problem'],
            ground_truth=result['ground_truth'],
            model_answer=result['model_answer'],
        )

        # Get judgment from Gemini
        try:
            judgment = client.generate(
                model_id="gemini-2.0-flash-exp",  # Best Gemini model
                prompt=prompt,
                temperature=0.0,  # Deterministic
                max_tokens=10,  # Only need "Correct" or "Incorrect"
            ).strip()

            # Normalize judgment
            is_correct = "correct" in judgment.lower()

        except Exception as e:
            print(f"\n  Error judging example {i}: {e}")
            judgment = "Error"
            is_correct = False

        # Add judgment to result
        judged_result = {
            **result,
            'judgment': judgment,
            'is_correct': is_correct,
        }

        judged_results.append(judged_result)

        # Periodic progress
        if (i + 1) % 100 == 0:
            correct_so_far = sum(r['is_correct'] for r in judged_results)
            accuracy_so_far = correct_so_far / len(judged_results)
            print(f"\n  Progress: {i+1}/{len(results)} | Accuracy so far: {accuracy_so_far:.1%}")

    # Save judged results
    print(f"\nSaving judged results to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for result in judged_results:
            f.write(json.dumps(result) + '\n')

    # Calculate final statistics
    num_correct = sum(r['is_correct'] for r in judged_results)
    accuracy = num_correct / len(judged_results)

    print(f"\n{'='*80}")
    print("JUDGING COMPLETE")
    print(f"{'='*80}")
    print(f"Total examples: {len(judged_results)}")
    print(f"Correct: {num_correct}")
    print(f"Incorrect: {len(judged_results) - num_correct}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Results saved to: {output_file}")

    return judged_results


def main():
    parser = argparse.ArgumentParser(
        description="Judge SLM answers using Gemini Pro"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input SLM results JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output judged results JSONL file"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for API calls"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Must provide --api-key or set GEMINI_API_KEY env var")

    # Judge answers
    judge_answers(
        input_file=Path(args.input),
        output_file=Path(args.output),
        gemini_api_key=api_key,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
