"""
Modal script to evaluate RL-calibrated model on test set.
Computes calibration metrics and confidence distribution.
"""

import modal
import json

app = modal.App("evaluate-rl-calibration")

# Image with evaluation dependencies
eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",
        "scipy",
        "torch==2.4.0",
        "transformers==4.45.2",
        "peft==0.13.0",
        "scikit-learn",
    )
)

volume = modal.Volume.from_name("slm-dpo-training", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(volumes={VOLUME_PATH: volume})
def upload_test_data(content: str, path: str):
    """Upload test data to Modal volume."""
    from pathlib import Path

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    volume.commit()
    print(f"Test data uploaded to {path}")


@app.function(
    image=eval_image,
    gpu="A100-40GB",
    timeout=1800,  # 30 minutes
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def evaluate_calibration(
    model_path: str,
    test_data_path: str,
    output_path: str,
):
    """
    Evaluate calibration on test set.

    Computes:
    - RMS Calibration Error
    - Expected Calibration Error (ECE)
    - Confidence distribution
    - Accuracy by confidence bin
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import numpy as np
    from sklearn.metrics import mean_squared_error
    import re
    from pathlib import Path

    print(f"\n{'='*80}")
    print(f"RL Calibration Evaluation")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"{'='*80}\n")

    # Load test data
    print(f"Loading test data from {test_data_path}")
    test_examples = []
    with open(test_data_path, 'r') as f:
        for line in f:
            test_examples.append(json.loads(line))

    print(f"Loaded {len(test_examples)} test examples")

    # Load base model
    print(f"\nLoading base model...")
    base_model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load RL-trained adapter
    print(f"Loading RL adapter from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    print(f"Model loaded successfully!")

    # Evaluate
    print(f"\n{'='*80}")
    print(f"Running evaluation on {len(test_examples)} examples")
    print(f"{'='*80}\n")

    predictions = []
    confidences = []
    ground_truth = []

    for i, ex in enumerate(test_examples):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(test_examples)}")

        question = ex['problem']
        is_correct = ex['is_correct']

        # Format prompt
        prompt = f"Question: {question}\n\nProvide your answer and confidence (0.0-1.0):"

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        # Extract confidence
        try:
            confidence_match = re.search(r'[Cc]onfidence[:\s]+([0-9.]+)', response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5
        except:
            confidence = 0.5

        predictions.append(response)
        confidences.append(confidence)
        ground_truth.append(1.0 if is_correct else 0.0)

    # Compute metrics
    print(f"\n{'='*80}")
    print(f"Computing Calibration Metrics")
    print(f"{'='*80}\n")

    confidences = np.array(confidences)
    ground_truth = np.array(ground_truth)

    # 1. RMS Calibration Error
    rms_error = np.sqrt(mean_squared_error(ground_truth, confidences))

    # 2. Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = ground_truth[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(None)
            bin_confs.append(None)
            bin_counts.append(0)

    # 3. Confidence distribution stats
    conf_mean = confidences.mean()
    conf_std = confidences.std()
    conf_min = confidences.min()
    conf_max = confidences.max()

    # 4. Accuracy
    accuracy = ground_truth.mean()

    # Print results
    print(f"Overall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  RMS Calibration Error: {rms_error:.4f}")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"\nConfidence Distribution:")
    print(f"  Mean: {conf_mean:.4f}")
    print(f"  Std: {conf_std:.4f}")
    print(f"  Range: [{conf_min:.4f}, {conf_max:.4f}]")
    print(f"\nCalibration by Confidence Bin:")
    for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if bin_counts[i] > 0:
            print(f"  [{lower:.1f}, {upper:.1f}]: "
                  f"Acc={bin_accs[i]:.3f}, "
                  f"Conf={bin_confs[i]:.3f}, "
                  f"Count={bin_counts[i]}")

    # Save results
    results = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "n_examples": len(test_examples),
        "metrics": {
            "accuracy": float(accuracy),
            "rms_calibration_error": float(rms_error),
            "expected_calibration_error": float(ece),
        },
        "confidence_stats": {
            "mean": float(conf_mean),
            "std": float(conf_std),
            "min": float(conf_min),
            "max": float(conf_max),
        },
        "calibration_bins": [
            {
                "bin": f"[{lower:.1f}, {upper:.1f}]",
                "accuracy": float(bin_accs[i]) if bin_accs[i] is not None else None,
                "avg_confidence": float(bin_confs[i]) if bin_confs[i] is not None else None,
                "count": int(bin_counts[i]),
            }
            for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers))
        ],
    }

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    volume.commit()

    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

    return results


@app.local_entrypoint()
def main(
    model_path: str = None,
    model_type: str = "rl",  # "rl" or "grpo"
    test_data: str = "data/slm_baseline/google_gemma-2-2b-it_judged.jsonl",
):
    """
    Evaluate RL-calibrated model (REINFORCE or GRPO).

    Usage:
        # Evaluate REINFORCE model
        modal run scripts/modal_evaluate_rl_calibration.py

        # Evaluate GRPO model
        modal run scripts/modal_evaluate_rl_calibration.py --model-type grpo
    """

    # Default to appropriate model path based on type
    if model_path is None:
        if model_type == "grpo":
            model_path = f"{VOLUME_PATH}/grpo_calibrated_models/google_gemma-2-2b-it_grpo_vllm"
        else:  # default to rl (REINFORCE)
            model_path = f"{VOLUME_PATH}/rl_calibrated_models/google_gemma-2-2b-it_rl_vllm"

    print(f"{'='*80}")
    print(f"RL CALIBRATION EVALUATION ({model_type.upper()})")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Test data: {test_data}")
    print(f"{'='*80}\n")

    # Upload test data
    print(f"Uploading test data...")
    with open(test_data, 'r') as f:
        test_content = f.read()

    remote_test_path = f"{VOLUME_PATH}/test_data/google_gemma-2-2b-it_test.jsonl"

    upload_test_data.remote(test_content, remote_test_path)

    # Run evaluation
    output_path = f"{VOLUME_PATH}/evaluations/rl_calibration_results.json"

    print(f"\nRunning evaluation...")
    results = evaluate_calibration.remote(
        model_path=model_path,
        test_data_path=remote_test_path,
        output_path=output_path,
    )

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nKey Metrics:")
    print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"  RMS Calibration Error: {results['metrics']['rms_calibration_error']:.4f}")
    print(f"  ECE: {results['metrics']['expected_calibration_error']:.4f}")
    print(f"  Confidence: {results['confidence_stats']['mean']:.4f} ± {results['confidence_stats']['std']:.4f}")
    print(f"\nResults saved to: {output_path}")
    print(f"\nTo download results:")
    print(f"  modal volume get slm-dpo-training {output_path}")
