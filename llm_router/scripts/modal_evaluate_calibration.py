"""
Modal script for evaluating calibrated models.
Separates evaluation from training for easier iteration and debugging.
"""

import modal
import json
from pathlib import Path

app = modal.App("slm-calibration-eval")

# Evaluation image
eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",
        "torch==2.1.0",
        "transformers==4.42.0",
        "peft==0.11.0",
        "datasets==2.15.0",
        "accelerate==0.30.0",
        "bitsandbytes==0.43.0",
        "tqdm",
    )
)

volume_dpo = modal.Volume.from_name("slm-dpo-training", create_if_missing=True)
volume_implicit = modal.Volume.from_name("slm-implicit-calibration", create_if_missing=True)
volume_sft = modal.Volume.from_name("slm-sft-calibration", create_if_missing=True)
VOLUME_PATH = "/vol"


def get_volume_for_model(model_path: str):
    """Determine which volume to use based on model path."""
    if "dpo_models" in model_path:
        return volume_dpo
    elif "implicit" in model_path:
        return volume_implicit
    elif "sft" in model_path:
        return volume_sft
    else:
        # Default to implicit
        return volume_implicit

# Note: We'll set the volume dynamically in the main function
# For now, use implicit as default
@app.function(
    image=eval_image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={VOLUME_PATH: volume_dpo},  # Mount DPO volume (SFT model saved here)
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def evaluate_calibrated_model(
    model_path: str,
    test_data_path: str,
    output_path: str,
    use_lora: bool = True,
    model_type: str = "implicit",  # "dpo", "implicit", or "sft"
):
    """
    Evaluate a calibrated model and calculate RMS calibration error.

    Args:
        model_path: Path to trained model (on Modal volume)
        test_data_path: Path to test data with ground truth
        output_path: Where to save evaluation results
        use_lora: Whether model uses LoRA adapters
        model_type: "dpo" (explicit confidence), "implicit" (logprobs), or "sft"
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import numpy as np
    from tqdm import tqdm

    print(f"\n{'='*80}")
    print(f"Evaluating Calibrated Model: {model_path}")
    print(f"{'='*80}\n")

    # Load test data
    print(f"Loading test data from {test_data_path}")
    test_examples = []
    with open(test_data_path, 'r') as f:
        for line in f:
            test_examples.append(json.loads(line))

    print(f"Loaded {len(test_examples)} test examples")

    # Load model and tokenizer
    if use_lora:
        # Load base model + adapter
        base_model_name = "google/gemma-2-2b-it"  # TODO: Make configurable
        print(f"\nLoading tokenizer from base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        print(f"\nLoading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()

    # Run inference
    print(f"\nRunning inference on {len(test_examples)} examples...")
    results = []

    for ex in tqdm(test_examples):
        question = ex["problem"]
        ground_truth = ex["is_correct"]

        # Generate answer
        prompt = f"Question: {question}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            if model_type == "dpo":
                # DPO: Extract confidence from generated text
                # Disable caching to avoid cache_position bugs in transformers 4.42
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False,  # Disable KV cache to avoid past_length=None bug
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Parse confidence from "Confidence: X" format
                confidence = None
                if "Confidence:" in generated_text:
                    try:
                        conf_str = generated_text.split("Confidence:")[1].strip().split()[0]
                        confidence = float(conf_str)
                    except:
                        confidence = 0.5  # Default if parsing fails
                else:
                    confidence = 0.5  # Default if no confidence found

            else:  # implicit
                # Implicit: Generate answer first, then compute confidence from generated token logprobs
                # Disable caching to avoid cache_position bugs
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Extract generated tokens (exclude input)
                input_length = inputs["input_ids"].shape[1]
                gen_tokens = generated_ids.sequences[0, input_length:]

                # Get scores for generated tokens
                scores = generated_ids.scores  # List of tensors, one per generated token

                if len(scores) > 0:
                    # Compute confidence as geometric mean of token probabilities
                    token_probs = []
                    for i, score_tensor in enumerate(scores):
                        # score_tensor shape: (1, vocab_size)
                        probs = F.softmax(score_tensor[0], dim=-1)
                        # Get probability of actual generated token
                        if i < len(gen_tokens):
                            token_id = gen_tokens[i].item()
                            token_prob = probs[token_id].item()
                            token_probs.append(token_prob)

                    if token_probs:
                        # Geometric mean
                        log_probs = [np.log(p + 1e-10) for p in token_probs]
                        avg_log_prob = np.mean(log_probs)
                        confidence = np.exp(avg_log_prob)
                    else:
                        confidence = 0.5  # Default if no tokens generated
                else:
                    confidence = 0.5  # Default if no generation

        results.append({
            "problem": question,
            "is_correct": ground_truth,
            "confidence": confidence,
        })

    # Calculate calibration metrics
    print("\nCalculating calibration metrics...")

    # RMS calibration error (HLE with β=100)
    def calculate_rms_error(results, beta=100):
        """Calculate RMS calibration error using binning."""
        confidences = np.array([r["confidence"] for r in results])
        correctness = np.array([r["is_correct"] for r in results])

        # Create bins
        bin_edges = np.arange(0, 1.01, 1/beta)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        squared_errors = []
        bin_stats = []

        for i in range(len(bin_edges) - 1):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = correctness[mask].mean()
                bin_count = mask.sum()

                squared_error = (bin_conf - bin_acc) ** 2
                squared_errors.append(squared_error)

                bin_stats.append({
                    "bin": f"[{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}]",
                    "count": int(bin_count),
                    "avg_confidence": float(bin_conf),
                    "accuracy": float(bin_acc),
                    "error": float(np.sqrt(squared_error)),
                })

        rms_error = np.sqrt(np.mean(squared_errors))
        return rms_error, bin_stats

    rms_error, bin_stats = calculate_rms_error(results)

    # Overall stats
    avg_confidence = np.mean([r["confidence"] for r in results])
    accuracy = np.mean([r["is_correct"] for r in results])

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total examples: {len(results)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"RMS Calibration Error: {rms_error:.4f}")
    print(f"\nTop 5 miscalibrated bins:")
    sorted_bins = sorted(bin_stats, key=lambda x: x["error"], reverse=True)[:5]
    for b in sorted_bins:
        print(f"  {b['bin']}: conf={b['avg_confidence']:.3f}, acc={b['accuracy']:.3f}, error={b['error']:.3f} (n={b['count']})")

    # Save results
    output_data = {
        "model_path": model_path,
        "model_type": model_type,
        "total_examples": len(results),
        "accuracy": float(accuracy),
        "avg_confidence": float(avg_confidence),
        "rms_calibration_error": float(rms_error),
        "bin_stats": bin_stats,
        "per_example_results": results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    volume_dpo.commit()

    print(f"\nResults saved to {output_path}")
    print(f"{'='*80}\n")

    return {
        "rms_error": rms_error,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
    }


@app.function(volumes={VOLUME_PATH: volume_dpo})
def upload_test_data(content: str, remote_path: str):
    """Upload test data to Modal volume."""
    from pathlib import Path

    # Create parent directory if it doesn't exist
    remote_file = Path(remote_path)
    remote_file.parent.mkdir(parents=True, exist_ok=True)

    with open(remote_path, 'w') as f:
        f.write(content)
    volume_dpo.commit()
    print(f"Test data uploaded to {remote_path}")


@app.local_entrypoint()
def main(
    model_path: str,
    test_data: str = "data/slm_baseline/google_gemma-2-2b-it_judged.jsonl",
    model_type: str = "dpo",
):
    """
    Evaluate a calibrated model.

    Usage:
        # Test with 10 examples
        modal run scripts/modal_evaluate_calibration.py \\
            --model-path /vol/dpo_models/google_gemma-2-2b-it_calibrated \\
            --test-data data/slm_baseline/test_small.jsonl \\
            --model-type dpo

        # Full evaluation
        modal run scripts/modal_evaluate_calibration.py \\
            --model-path /vol/dpo_models/google_gemma-2-2b-it_calibrated \\
            --model-type dpo
    """
    from pathlib import Path

    print(f"{'='*80}")
    print("CALIBRATION EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Test data: {test_data}")
    print()

    # Upload test data to Modal
    print(f"Uploading test data: {test_data}")
    with open(test_data, 'r') as f:
        test_content = f.read()

    test_remote_path = f"{VOLUME_PATH}/eval_data/test_data.jsonl"
    upload_test_data.remote(test_content, test_remote_path)

    # Run evaluation
    output_path = f"{VOLUME_PATH}/eval_results/{Path(model_path).name}_eval.json"

    print(f"\nStarting evaluation...")
    result = evaluate_calibrated_model.remote(
        model_path=model_path,
        test_data_path=test_remote_path,
        output_path=output_path,
        model_type=model_type,
    )

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"RMS Calibration Error: {result['rms_error']:.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Average Confidence: {result['avg_confidence']:.4f}")
    print(f"\nResults saved to: {output_path}")
    print(f"\nTo download results:")
    print(f"  modal volume get slm-dpo-training {output_path}")
