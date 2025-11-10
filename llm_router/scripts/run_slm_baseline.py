#!/usr/bin/env python3
"""
Run SLM baseline evaluation on SimpleQA-Verified using vLLM.
Measures calibration of SLMs of different sizes.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# vLLM imports
from vllm import LLM, SamplingParams


def load_simpleqa_verified(file_path: str, limit: int = None) -> List[Dict]:
    """Load SimpleQA-Verified dataset."""
    examples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            examples.append(json.loads(line))
    return examples


def create_qa_prompt(problem: str) -> str:
    """
    Create prompt for SLM to answer question.

    Format: Question → Answer directly (no confidence in prompt, we extract from logprobs)
    """
    return f"""Answer the following question concisely and accurately.

Question: {problem}

Answer:"""


def extract_confidence_from_logprobs(logprobs_data: Any) -> float:
    """
    Extract confidence score from vLLM logprobs using entropy.

    Method: Average entropy across generated tokens.
    Lower entropy = higher confidence (model is more certain).

    Entropy formula: H = -Σ p(x) * log(p(x))
    - H = 0: Completely certain (one token has prob = 1)
    - H = log(V): Maximum uncertainty (uniform over vocabulary)

    We convert to confidence: confidence = 1 - (H / H_max)
    where H_max is normalized entropy.

    Args:
        logprobs_data: vLLM output logprobs (list of dicts with top-k tokens)

    Returns:
        Confidence score in [0, 1]
    """
    if not logprobs_data:
        return 0.5  # Default uncertainty

    entropies = []

    for token_data in logprobs_data:
        # Each token_data contains logprobs for top-k tokens
        # We need the full distribution, but vLLM only gives top-k
        # So we compute entropy over top-k (approximation)

        if not hasattr(token_data, 'logprob'):
            continue

        # Get top-k logprobs (vLLM provides dict of {token_id: logprob})
        # For the selected token, we at least have its logprob
        selected_logprob = token_data.logprob
        selected_prob = np.exp(selected_logprob)

        # If we only have the selected token, assume binary distribution
        # (selected vs "everything else")
        # This is a simplification but works well in practice
        other_prob = 1.0 - selected_prob

        # Compute entropy: H = -Σ p * log(p)
        entropy = 0.0
        if selected_prob > 0:
            entropy -= selected_prob * selected_logprob  # p * log(p), logprob already in log space
        if other_prob > 0:
            entropy -= other_prob * np.log(other_prob)

        entropies.append(entropy)

    if not entropies:
        return 0.5

    # Average entropy across all tokens
    avg_entropy = np.mean(entropies)

    # Normalize entropy to [0, 1]
    # Max entropy for binary distribution is log(2) ≈ 0.693
    max_entropy = np.log(2)
    normalized_entropy = avg_entropy / max_entropy

    # Convert to confidence: low entropy = high confidence
    confidence = 1.0 - np.clip(normalized_entropy, 0.0, 1.0)

    return float(confidence)


def run_inference(
    model_name: str,
    examples: List[Dict],
    output_file: Path,
    max_tokens: int = 100,
    temperature: float = 0.0,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Run vLLM inference on SimpleQA examples.

    Args:
        model_name: HuggingFace model ID
        examples: List of SimpleQA examples
        output_file: Where to save results
        max_tokens: Max generation length
        temperature: Sampling temperature (0 = greedy)
        batch_size: Batch size for inference

    Returns:
        List of results with answers and confidence scores
    """
    print(f"\n{'='*80}")
    print(f"Running inference: {model_name}")
    print(f"{'='*80}")
    print(f"Examples: {len(examples)}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Initialize vLLM
    print("Loading model with vLLM...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Adjust based on GPU availability
        gpu_memory_utilization=0.9,
        max_model_len=2048,  # Sufficient for SimpleQA
    )

    # Sampling parameters with logprobs enabled
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=1,  # Request logprobs for confidence extraction
    )

    # Prepare prompts
    prompts = [create_qa_prompt(ex['problem']) for ex in examples]

    print(f"Running inference on {len(prompts)} examples...")

    # Run batch inference
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    results = []
    for i, (example, output) in enumerate(zip(examples, outputs)):
        generated_text = output.outputs[0].text.strip()
        logprobs = output.outputs[0].logprobs

        # Extract confidence from logprobs
        confidence = extract_confidence_from_logprobs(logprobs)

        result = {
            'original_index': example['original_index'],
            'problem': example['problem'],
            'ground_truth': example['answer'],
            'model_answer': generated_text,
            'confidence': confidence,
            'topic': example['topic'],
            'answer_type': example['answer_type'],
        }

        results.append(result)

        # Periodic progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(examples)} examples")

    # Save results
    print(f"\nSaving results to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✅ Inference complete!")
    print(f"   Total examples: {len(results)}")
    print(f"   Average confidence: {np.mean([r['confidence'] for r in results]):.3f}")
    print(f"   Confidence range: [{np.min([r['confidence'] for r in results]):.3f}, "
          f"{np.max([r['confidence'] for r in results]):.3f}]")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run SLM baseline evaluation with vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., google/gemma-2-2b-it)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmarks/simpleqa_verified.jsonl",
        help="Path to SimpleQA-Verified dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/slm_baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    examples = load_simpleqa_verified(args.dataset, limit=args.limit)
    print(f"Loaded {len(examples)} examples")

    # Create output filename
    model_name_safe = args.model.replace('/', '_')
    output_file = Path(args.output_dir) / f"{model_name_safe}_results.jsonl"

    # Run inference
    results = run_inference(
        model_name=args.model,
        examples=examples,
        output_file=output_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Next step: Run evaluation/judge to calculate correctness")


if __name__ == "__main__":
    main()
