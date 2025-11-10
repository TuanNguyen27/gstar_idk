#!/usr/bin/env python3
"""
Fast async judge for SLM answers using Gemini Pro with concurrent requests.
Optimized for paid tier rate limits (1500+ RPM).
"""

import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict
import sys
import os
from tqdm.asyncio import tqdm_asyncio
import google.generativeai as genai

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_judge_prompt(question: str, ground_truth: str, model_answer: str) -> str:
    """Create prompt for Gemini judge."""
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


async def judge_single(
    model: genai.GenerativeModel,
    result: Dict,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Judge a single answer with rate limiting."""
    async with semaphore:
        try:
            prompt = create_judge_prompt(
                question=result['problem'],
                ground_truth=result['ground_truth'],
                model_answer=result['model_answer'],
            )

            # Generate judgment
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=10,
                ),
            )

            judgment = response.text.strip()
            is_correct = "correct" in judgment.lower()

        except Exception as e:
            print(f"\n  Error: {e}")
            judgment = "Error"
            is_correct = False

        return {
            **result,
            'judgment': judgment,
            'is_correct': is_correct,
        }


async def judge_answers_async(
    input_file: Path,
    output_file: Path,
    gemini_api_key: str,
    max_concurrent: int = 100,  # Paid tier supports high concurrency
) -> List[Dict]:
    """
    Judge SLM answers using async Gemini Pro with concurrency.

    Args:
        input_file: Path to SLM results JSONL
        output_file: Path to save judged results
        gemini_api_key: Gemini API key
        max_concurrent: Maximum concurrent requests

    Returns:
        List of judged results
    """
    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    # Load results
    print(f"Loading results from {input_file}")
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(results)} examples")
    print(f"Judging with Gemini Pro (max {max_concurrent} concurrent requests)...")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Judge all answers concurrently
    tasks = [judge_single(model, result, semaphore) for result in results]
    judged_results = await tqdm_asyncio.gather(*tasks, desc="Judging")

    # Save results
    print(f"\nSaving judged results to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for result in judged_results:
            f.write(json.dumps(result) + '\n')

    # Calculate statistics
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
        description="Fast async judge for SLM answers"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-concurrent", type=int, default=100,
                        help="Max concurrent requests (paid tier: 100-200)")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Must provide --api-key or set GEMINI_API_KEY")

    # Run async judging
    asyncio.run(judge_answers_async(
        input_file=Path(args.input),
        output_file=Path(args.output),
        gemini_api_key=api_key,
        max_concurrent=args.max_concurrent,
    ))


if __name__ == "__main__":
    main()
