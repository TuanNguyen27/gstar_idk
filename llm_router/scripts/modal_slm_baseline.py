"""
Modal script for running SLM baseline inference with vLLM.
Measures calibration of SLMs across different sizes.
"""

import modal
import json
from pathlib import Path

# Create Modal app
app = modal.App("slm-baseline-inference")

# Define Modal image with vLLM
vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.6.3",
        "numpy",
        "tqdm",
    )
)

# Create volume for storing results
volume = modal.Volume.from_name("slm-baseline-results", create_if_missing=True)

VOLUME_PATH = "/vol"


@app.function(
    image=vllm_image,
    gpu="A100-40GB",  # A100 40GB for larger models
    timeout=3600 * 2,  # 2 hours
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_slm_inference(
    model_name: str,
    dataset_path: str,
    limit: int = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
):
    """
    Run vLLM inference on SimpleQA-Verified dataset.

    Args:
        model_name: HuggingFace model ID
        dataset_path: Path to SimpleQA-Verified JSONL
        limit: Limit number of examples (for testing)
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
    """
    import numpy as np
    import os
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    print(f"\n{'='*80}")
    print(f"SLM Baseline Inference: {model_name}")
    print(f"{'='*80}\n")

    # Verify HF token is available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("HuggingFace token detected ✓")
    else:
        print("WARNING: No HF_TOKEN found, may fail for gated models")

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    examples = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Initialize vLLM
    print(f"\nInitializing vLLM with model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        trust_remote_code=True,  # For some models like Qwen
        download_dir="/tmp/hf_cache",  # Use temp dir for caching
    )

    # Sampling parameters with logprobs
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=1,  # Enable logprobs for confidence extraction
    )

    # Prepare prompts
    def create_qa_prompt(problem: str) -> str:
        """Create QA prompt."""
        return f"""Answer the following question concisely and accurately.

Question: {problem}

Answer:"""

    prompts = [create_qa_prompt(ex['problem']) for ex in examples]

    # Run inference
    print(f"\nRunning inference on {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params)

    # Extract confidence using entropy
    def extract_confidence_from_logprobs(logprobs_data):
        """
        Extract confidence from logprobs using entropy.

        Lower entropy = higher confidence.
        """
        if not logprobs_data:
            print("  WARNING: No logprobs data")
            return 0.5

        # vLLM returns a list of dicts, where each dict maps token_id -> Logprob object
        probabilities = []

        for token_logprobs in logprobs_data:
            if token_logprobs is None:
                continue

            # token_logprobs is a dict: {token_id: Logprob(logprob=..., rank=..., ...)}
            # We want the logprob of the selected (top-1) token
            if isinstance(token_logprobs, dict) and len(token_logprobs) > 0:
                # Get the highest-ranked token (rank=1 is the selected token)
                selected_logprob = None
                for token_id, logprob_obj in token_logprobs.items():
                    if hasattr(logprob_obj, 'rank') and logprob_obj.rank == 1:
                        selected_logprob = logprob_obj.logprob
                        break
                    elif hasattr(logprob_obj, 'logprob'):
                        # Fallback: take first one if no rank
                        selected_logprob = logprob_obj.logprob
                        break

                if selected_logprob is not None:
                    prob = np.exp(selected_logprob)
                    probabilities.append(prob)

        if not probabilities:
            print("  WARNING: Could not extract probabilities from logprobs")
            return 0.5

        # Compute average probability (simple confidence measure)
        # Higher average probability = higher confidence
        avg_prob = np.mean(probabilities)

        return float(avg_prob)

    # Process results
    results = []
    print("\nProcessing results...")

    for i, (example, output) in enumerate(tqdm(zip(examples, outputs), total=len(examples))):
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

    # Save results
    model_name_safe = model_name.replace('/', '_')
    output_file = f"{VOLUME_PATH}/{model_name_safe}_results.jsonl"

    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Commit volume
    volume.commit()

    # Print statistics
    confidences = [r['confidence'] for r in results]
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Total examples: {len(results)}")
    print(f"Average confidence: {np.mean(confidences):.3f}")
    print(f"Confidence std: {np.std(confidences):.3f}")
    print(f"Confidence range: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")
    print(f"Results saved to: {output_file}")

    return {
        'model': model_name,
        'num_examples': len(results),
        'avg_confidence': float(np.mean(confidences)),
        'output_file': output_file,
    }


@app.function(volumes={VOLUME_PATH: volume})
def upload_dataset(content: str, dataset_remote_path: str):
    """Upload dataset to Modal volume."""
    with open(dataset_remote_path, 'w') as f:
        f.write(content)
    volume.commit()
    print(f"Dataset uploaded to {dataset_remote_path}")


@app.local_entrypoint()
def main(
    model: str = "google/gemma-2-2b-it",
    dataset: str = "data/benchmarks/simpleqa_verified.jsonl",
    limit: int = None,
):
    """
    Main entrypoint for running SLM baseline inference.

    Usage:
        modal run scripts/modal_slm_baseline.py --model google/gemma-2-2b-it --limit 100

    Models to test:
        - google/gemma-2-9b-it
        - google/gemma-2-2b-it
        - Qwen/Qwen2.5-1.5B-Instruct
    """
    # Upload dataset to Modal if needed
    print(f"Uploading dataset: {dataset}")

    # Read local dataset
    with open(dataset, 'r') as f:
        dataset_content = f.read()

    # Create temp file in Modal volume
    dataset_remote_path = f"{VOLUME_PATH}/simpleqa_verified.jsonl"

    # Write dataset to volume
    upload_dataset.remote(dataset_content, dataset_remote_path)

    # Run inference
    print(f"\nStarting inference for: {model}")
    result = run_slm_inference.remote(
        model_name=model,
        dataset_path=dataset_remote_path,
        limit=limit,
    )

    print(f"\n{'='*80}")
    print("JOB COMPLETE")
    print(f"{'='*80}")
    print(f"Model: {result['model']}")
    print(f"Examples processed: {result['num_examples']}")
    print(f"Average confidence: {result['avg_confidence']:.3f}")
    print(f"\nResults saved to Modal volume at:")
    print(f"  {result['output_file']}")
    print(f"\nTo download results:")
    print(f"  modal volume get slm-baseline-results {result['output_file']}")
