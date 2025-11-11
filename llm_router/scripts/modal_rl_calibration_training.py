"""
Modal script for RL-based calibration training with anti-collapse rewards.
Uses REINFORCE (policy gradient) with sharpness penalties to avoid mode collapse.
"""

import modal
import json
from pathlib import Path

app = modal.App("slm-rl-calibration-training")

# Training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",
        "scipy",
        "torch==2.1.0",
        "transformers==4.42.0",
        "peft==0.11.0",
        "datasets==2.15.0",
        "accelerate==0.30.0",
        "bitsandbytes==0.43.0",
        "tqdm",
    )
)

volume = modal.Volume.from_name("slm-dpo-training", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(
    image=training_image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_rl_calibration_model(
    model_name: str,
    judged_data_path: str,
    output_path: str,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lambda_sharp: float = 0.3,
    lambda_extreme: float = 0.2,
):
    """
    Train calibration model using RL (REINFORCE) with anti-collapse rewards.

    Args:
        model_name: Base model to fine-tune
        judged_data_path: Path to judged data with correctness labels
        output_path: Where to save trained model
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        lambda_sharp: Weight for sharpness bonus (anti-0.5 penalty)
        lambda_extreme: Weight for extremity bonus (prefer edges)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    import numpy as np
    from tqdm import tqdm
    import re

    print(f"\n{'='*80}")
    print(f"RL-Based Calibration Training (REINFORCE)")
    print(f"{'='*80}")
    print(f"Base model: {model_name}")
    print(f"Training data: {judged_data_path}")
    print(f"Output: {output_path}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Anti-collapse params:")
    print(f"  lambda_sharp: {lambda_sharp} (sharpness bonus)")
    print(f"  lambda_extreme: {lambda_extreme} (extremity bonus)")
    print(f"{'='*80}\n")

    # Load judged data
    print(f"Loading judged data from {judged_data_path}")
    examples = []
    with open(judged_data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")
    correct_count = sum(1 for ex in examples if ex['is_correct'])
    print(f"  Correct: {correct_count} ({correct_count/len(examples)*100:.1f}%)")
    print(f"  Incorrect: {len(examples)-correct_count} ({(len(examples)-correct_count)/len(examples)*100:.1f}%)")

    # Load tokenizer and model
    print(f"\nLoading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Confidence history for diversity tracking
    confidence_history = []

    # Reward function with anti-collapse mechanisms
    def compute_rl_reward(confidence: float, is_correct: bool) -> float:
        """
        RL reward with multiple anti-collapse mechanisms:
        1. Base calibration reward (negative squared error)
        2. Sharpness bonus (penalize 0.5)
        3. Extremity bonus (reward edges)
        4. Diversity bonus (reward variance)
        """
        # Component 1: Calibration accuracy
        target = 1.0 if is_correct else 0.0
        r_calib = -(confidence - target) ** 2

        # Component 2: Sharpness (distance from 0.5)
        r_sharp = abs(confidence - 0.5)

        # Component 3: Extremity bonus (prefer 0.0-0.3 or 0.7-1.0)
        if confidence < 0.3 or confidence > 0.7:
            r_extreme = 1.0
        else:
            r_extreme = -0.5  # Penalty for middle range

        # Component 4: Diversity bonus (variance of recent confidences)
        confidence_history.append(confidence)
        if len(confidence_history) > 100:
            confidence_history.pop(0)

        if len(confidence_history) > 10:
            hist_std = np.std(confidence_history)
            r_diversity = hist_std  # Reward variance
        else:
            r_diversity = 0.0

        # Combined reward
        reward = (r_calib +
                  lambda_sharp * r_sharp +
                  lambda_extreme * r_extreme +
                  0.1 * r_diversity)

        return reward, {
            'calib': r_calib,
            'sharp': r_sharp,
            'extreme': r_extreme,
            'diversity': r_diversity,
        }

    # REINFORCE training loop
    print("\n" + "="*80)
    print("Starting RL Training (REINFORCE)")
    print("="*80 + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    total_steps = (len(examples) // batch_size) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()

    global_step = 0
    epoch_rewards = []
    epoch_confidences = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Shuffle examples
        import random
        random.shuffle(examples)

        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_conf = []

        for i in tqdm(range(0, len(examples), batch_size)):
            batch = examples[i:i+batch_size]

            batch_loss = 0.0
            batch_rewards = []
            batch_confidences = []

            for ex in batch:
                question = ex['problem']  # Field is 'problem', not 'question'
                answer = ex['model_answer']
                is_correct = ex['is_correct']

                # Format prompt
                prompt = f"Question: {question}\n\nProvide your answer and confidence (0.0-1.0):"

                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate with sampling
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                # Decode response
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()

                # Extract confidence (with error handling)
                try:
                    confidence_match = re.search(r'[Cc]onfidence[:\s]+([0-9.]+)', response)
                    if confidence_match:
                        confidence = float(confidence_match.group(1))
                        confidence = max(0.0, min(1.0, confidence))  # Clip to [0, 1]
                    else:
                        confidence = 0.5  # Default if not found
                except:
                    confidence = 0.5

                # Compute RL reward
                reward, reward_components = compute_rl_reward(confidence, is_correct)

                # Log probabilities for REINFORCE
                # Re-run forward pass to get log probs
                full_text = prompt + response
                full_inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}

                with torch.enable_grad():
                    model_outputs = model(**full_inputs, labels=full_inputs['input_ids'])
                    log_probs = -model_outputs.loss  # Negative log likelihood

                # REINFORCE loss: -log_prob * reward
                loss = -log_probs * reward

                batch_loss += loss
                batch_rewards.append(reward)
                batch_confidences.append(confidence)

            # Backward pass
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += batch_loss.item()
            epoch_reward += np.mean(batch_rewards)
            epoch_conf.extend(batch_confidences)

            global_step += 1

            # Logging
            if global_step % 10 == 0:
                avg_reward = np.mean(batch_rewards)
                avg_conf = np.mean(batch_confidences)
                conf_std = np.std(batch_confidences)
                print(f"Step {global_step}: reward={avg_reward:.4f}, conf={avg_conf:.3f}±{conf_std:.3f}")

        # Epoch statistics
        avg_epoch_loss = epoch_loss / (len(examples) // batch_size)
        avg_epoch_reward = epoch_reward / (len(examples) // batch_size)
        avg_epoch_conf = np.mean(epoch_conf)
        std_epoch_conf = np.std(epoch_conf)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_epoch_loss:.4f}")
        print(f"  Reward: {avg_epoch_reward:.4f}")
        print(f"  Confidence: {avg_epoch_conf:.3f} ± {std_epoch_conf:.3f}")
        print(f"  Confidence range: [{min(epoch_conf):.3f}, {max(epoch_conf):.3f}]")

        epoch_rewards.append(avg_epoch_reward)
        epoch_confidences.append((avg_epoch_conf, std_epoch_conf))

    # Save model
    print(f"\nSaving model to {output_path}")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save training statistics
    stats = {
        "epoch_rewards": epoch_rewards,
        "epoch_confidences": epoch_confidences,
        "final_confidence_mean": float(np.mean(epoch_conf)),
        "final_confidence_std": float(np.std(epoch_conf)),
        "lambda_sharp": lambda_sharp,
        "lambda_extreme": lambda_extreme,
    }

    with open(f"{output_path}/training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Commit volume
    volume.commit()

    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"Model saved to: {output_path}")
    print(f"Training stats saved to: {output_path}/training_stats.json")
    print(f"{'='*80}\n")

    return {"status": "success", "output_path": output_path, "stats": stats}


@app.function(volumes={VOLUME_PATH: volume})
def upload_judged_data(content: str, remote_path: str):
    """Upload judged data to Modal volume."""
    from pathlib import Path

    remote_file = Path(remote_path)
    remote_file.parent.mkdir(parents=True, exist_ok=True)

    with open(remote_path, 'w') as f:
        f.write(content)
    volume.commit()
    print(f"Judged data uploaded to {remote_path}")


@app.local_entrypoint()
def main(
    model: str = "google/gemma-2-2b-it",
    judged_data: str = "data/slm_baseline/google_gemma-2-2b-it_judged.jsonl",
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    lambda_sharp: float = 0.3,
    lambda_extreme: float = 0.2,
):
    """
    Train RL-based calibration model with anti-collapse rewards.

    Usage:
        modal run scripts/modal_rl_calibration_training.py \
            --model google/gemma-2-2b-it \
            --judged-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
            --learning-rate 1e-5 \
            --num-epochs 3 \
            --lambda-sharp 0.3 \
            --lambda-extreme 0.2
    """
    from pathlib import Path

    print(f"{'='*80}")
    print("RL CALIBRATION TRAINING (REINFORCE)")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Judged data: {judged_data}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Lambda sharp: {lambda_sharp}")
    print(f"Lambda extreme: {lambda_extreme}")
    print(f"{'='*80}\n")

    # Upload judged data to Modal
    print(f"Uploading judged data: {judged_data}")
    with open(judged_data, 'r') as f:
        judged_content = f.read()

    model_slug = model.replace("/", "_")
    remote_path = f"{VOLUME_PATH}/judged_data/{model_slug}_judged.jsonl"
    upload_judged_data.remote(judged_content, remote_path)

    # Train model
    output_path = f"{VOLUME_PATH}/rl_calibrated_models/{model_slug}_rl_calibrated"

    print(f"\nStarting RL training job...")
    result = train_rl_calibration_model.remote(
        model_name=model,
        judged_data_path=remote_path,
        output_path=output_path,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        lambda_sharp=lambda_sharp,
        lambda_extreme=lambda_extreme,
    )

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Status: {result['status']}")
    print(f"Model saved to: {result['output_path']}")
    print(f"\nTraining Statistics:")
    print(f"  Final confidence: {result['stats']['final_confidence_mean']:.3f} ± {result['stats']['final_confidence_std']:.3f}")
    print(f"  Epoch rewards: {result['stats']['epoch_rewards']}")
    print(f"\nTo download model:")
    print(f"  modal volume get slm-dpo-training {result['output_path']}")
