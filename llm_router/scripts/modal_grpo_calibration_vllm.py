"""
Modal script for GRPO-based calibration training using vLLM for inference.
Uses Group Relative Policy Optimization with anti-collapse rewards.
"""

import modal

app = modal.App("slm-grpo-calibration-vllm")

# vLLM image for fast inference
vllm_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.6.3.post1",
    "torch==2.4.0",
    "transformers==4.45.2",
)

# Training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",
        "scipy",
        "torch==2.4.0",
        "transformers==4.45.2",
        "peft==0.13.0",
        "datasets==2.15.0",
        "accelerate==0.34.0",
        "bitsandbytes==0.44.0",
        "tqdm",
    )
)

volume = modal.Volume.from_name("slm-dpo-training", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    timeout=600,
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def generate_rl_episodes(
    model_name: str,
    examples: list[dict],
    temperature: float = 0.7,
) -> list[dict]:
    """
    Generate episodes using vLLM for fast inference.

    Returns list of episodes with:
    - prompt
    - response
    - confidence
    - is_correct
    """
    from vllm import LLM, SamplingParams
    import re

    print(f"Loading model with vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    # Prepare prompts
    prompts = []
    for ex in examples:
        question = ex['problem']
        prompt = f"Question: {question}\n\nProvide your answer and confidence (0.0-1.0):"
        prompts.append(prompt)

    # Generate with sampling
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=100,
        top_p=0.95,
    )

    print(f"Generating {len(prompts)} responses with vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    # Extract episodes
    episodes = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()

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

        episodes.append({
            'prompt': prompts[i],
            'response': response,
            'confidence': confidence,
            'is_correct': examples[i]['is_correct'],
        })

    return episodes


@app.function(
    image=training_image,
    gpu="A100-40GB",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_grpo_calibration_model(
    model_name: str,
    judged_data_path: str,
    output_path: str,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    lambda_sharp: float = 0.3,
    lambda_extreme: float = 0.2,
    generation_batch_size: int = 100,
):
    """
    Train calibration model using GRPO with vLLM for generation.

    Key difference from REINFORCE: Rewards are normalized within each batch
    for better stability and variance reduction.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import numpy as np
    from tqdm import tqdm
    import json
    from pathlib import Path

    print(f"\n{'='*80}")
    print(f"GRPO-Based Calibration Training (GRPO + vLLM)")
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

    # Load tokenizer and model for training
    print(f"\nLoading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model {model_name} for training")
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
    def compute_rl_reward(confidence: float, is_correct: bool) -> tuple[float, dict]:
        """
        RL reward with multiple anti-collapse mechanisms.
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
            r_extreme = -0.5

        # Component 4: Diversity bonus
        confidence_history.append(confidence)
        if len(confidence_history) > 100:
            confidence_history.pop(0)

        if len(confidence_history) > 10:
            hist_std = np.std(confidence_history)
            r_diversity = hist_std
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

    # GRPO training loop
    print("\n" + "="*80)
    print("Starting GRPO Training (Group Relative Policy Optimization)")
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

        # Generate episodes in batches using vLLM
        print("Generating episodes with vLLM...")
        all_episodes = []
        for i in range(0, len(examples), generation_batch_size):
            batch_examples = examples[i:i+generation_batch_size]
            episodes = generate_rl_episodes.remote(
                model_name=model_name,
                examples=batch_examples,
                temperature=0.7,
            )
            all_episodes.extend(episodes)

        print(f"Generated {len(all_episodes)} episodes")

        # GRPO training loop on generated episodes
        for i in tqdm(range(0, len(all_episodes), batch_size)):
            episode_batch = all_episodes[i:i+batch_size]

            # Step 1: Collect all rewards and log_probs for the batch
            batch_rewards = []
            batch_log_probs = []
            batch_confidences = []

            for episode in episode_batch:
                # Compute reward
                reward, reward_components = compute_rl_reward(
                    episode['confidence'],
                    episode['is_correct']
                )

                # Compute log probabilities
                full_text = episode['prompt'] + episode['response']
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.enable_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    log_probs = -outputs.loss

                batch_rewards.append(reward)
                batch_log_probs.append(log_probs)
                batch_confidences.append(episode['confidence'])

            # Step 2: GRPO - Normalize rewards within batch
            mean_reward = np.mean(batch_rewards)
            std_reward = np.std(batch_rewards)
            normalized_rewards = [
                (r - mean_reward) / (std_reward + 1e-8)
                for r in batch_rewards
            ]

            # Step 3: Compute GRPO loss with normalized rewards
            batch_loss = 0.0
            for log_probs, norm_reward in zip(batch_log_probs, normalized_rewards):
                loss = -log_probs * norm_reward
                batch_loss += loss

            # Backward pass
            batch_loss = batch_loss / len(episode_batch)
            batch_loss.backward()

            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += batch_loss.item()
            epoch_reward += mean_reward  # Use raw mean reward for tracking
            epoch_conf.extend(batch_confidences)

            global_step += 1

            # Logging
            if global_step % 10 == 0:
                avg_reward = mean_reward
                avg_conf = np.mean(batch_confidences)
                conf_std = np.std(batch_confidences)
                print(f"Step {global_step}: reward={avg_reward:.4f}, conf={avg_conf:.3f}±{conf_std:.3f}")

        # Epoch statistics
        avg_epoch_loss = epoch_loss / (len(all_episodes) // batch_size)
        avg_epoch_reward = epoch_reward / (len(all_episodes) // batch_size)
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
    Train GRPO-based calibration model with vLLM for generation.

    Usage:
        modal run scripts/modal_grpo_calibration_vllm.py
    """
    from pathlib import Path

    print(f"{'='*80}")
    print("GRPO CALIBRATION TRAINING (Group Relative PO + vLLM)")
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
    output_path = f"{VOLUME_PATH}/grpo_calibrated_models/{model_slug}_grpo_vllm"

    print(f"\nStarting GRPO training job...")
    result = train_grpo_calibration_model.remote(
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
