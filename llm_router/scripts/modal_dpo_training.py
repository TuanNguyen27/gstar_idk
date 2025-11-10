"""
Modal script for DPO training to improve SLM calibration.
Trains models to output well-calibrated confidence scores.
"""

import modal
import json
from pathlib import Path

app = modal.App("slm-calibration-dpo")

# DPO training image with TRL
dpo_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",  # Pin numpy to <2 for compatibility
        "scipy",  # Required by bitsandbytes
        "torch==2.1.0",
        "transformers==4.42.0",  # Newer version for Gemma support
        "trl==0.8.6",  # For DPO training (DPOConfig added in 0.8.0+)
        "peft==0.11.0",  # For LoRA
        "datasets==2.15.0",
        "accelerate==0.30.0",
        "bitsandbytes==0.43.0",  # For quantization
        "wandb",  # For logging
    )
)

# Volume for storing training data and checkpoints
volume = modal.Volume.from_name("slm-dpo-training", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(
    image=dpo_image,
    gpu="A100-40GB",
    timeout=3600 * 4,  # 4 hours
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_dpo_calibration(
    model_name: str,
    dpo_data_path: str,
    output_dir: str,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    use_lora: bool = True,
):
    """
    Train SLM with DPO for calibration improvement.

    Args:
        model_name: HuggingFace model ID (e.g., "google/gemma-2-2b-it")
        dpo_data_path: Path to DPO preference pairs JSONL
        output_dir: Directory to save trained model
        learning_rate: Learning rate for DPO
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        use_lora: Whether to use LoRA (recommended for memory efficiency)
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from datasets import Dataset
    from trl import DPOTrainer
    from peft import LoraConfig, get_peft_model
    import wandb

    print(f"\n{'='*80}")
    print(f"DPO Calibration Training: {model_name}")
    print(f"{'='*80}\n")

    # Initialize wandb (optional)
    try:
        wandb.init(
            project="slm-calibration-dpo",
            name=f"{model_name.replace('/', '_')}_calibration",
            config={
                "model": model_name,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "use_lora": use_lora,
            }
        )
        use_wandb = True
    except:
        print("Wandb not available, skipping logging")
        use_wandb = False

    # Load DPO preference pairs
    print(f"Loading DPO data from {dpo_data_path}")
    dpo_examples = []
    with open(dpo_data_path, 'r') as f:
        for line in f:
            dpo_examples.append(json.loads(line))

    print(f"Loaded {len(dpo_examples)} DPO preference pairs")

    # Convert to HuggingFace Dataset format
    dataset_dict = {
        "prompt": [ex["prompt"] for ex in dpo_examples],
        "chosen": [ex["chosen"] for ex in dpo_examples],
        "rejected": [ex["rejected"] for ex in dpo_examples],
    }
    dataset = Dataset.from_dict(dataset_dict)

    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA if requested
    if use_lora:
        print("\nApplying LoRA configuration")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # DPO training config (using TrainingArguments for trl 0.8.6)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
    )

    # Initialize DPO Trainer
    print("\nInitializing DPO Trainer")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # DPO temperature parameter
    )

    # Train
    print("\nStarting DPO training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Commit volume
    volume.commit()

    print(f"\n{'='*80}")
    print("DPO TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model saved to: {output_dir}")

    return {
        "model": model_name,
        "output_dir": output_dir,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
    }


@app.function(volumes={VOLUME_PATH: volume})
def upload_dpo_data(content: str, remote_path: str):
    """Upload DPO training data to Modal volume."""
    from pathlib import Path

    # Create parent directory if it doesn't exist
    remote_file = Path(remote_path)
    remote_file.parent.mkdir(parents=True, exist_ok=True)

    with open(remote_path, 'w') as f:
        f.write(content)
    volume.commit()
    print(f"DPO data uploaded to {remote_path}")


@app.local_entrypoint()
def main(
    model: str = "google/gemma-2-2b-it",
    dpo_data: str = "data/slm_baseline/google_gemma-2-2b-it_dpo_pairs.jsonl",
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
):
    """
    Main entrypoint for DPO calibration training.

    Usage:
        modal run scripts/modal_dpo_training.py \\
            --model google/gemma-2-2b-it \\
            --dpo-data data/slm_baseline/google_gemma-2-2b-it_dpo_pairs.jsonl

    Models to train:
        - google/gemma-2-2b-it (2B params)
        - Qwen/Qwen2.5-1.5B-Instruct (1.5B params)
    """
    print(f"{'='*80}")
    print("DPO CALIBRATION TRAINING")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"DPO data: {dpo_data}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print()

    # Upload DPO data to Modal
    print(f"Uploading DPO data: {dpo_data}")
    with open(dpo_data, 'r') as f:
        dpo_content = f.read()

    model_safe_name = model.replace('/', '_')
    dpo_remote_path = f"{VOLUME_PATH}/dpo_data/{model_safe_name}_dpo_pairs.jsonl"
    upload_dpo_data.remote(dpo_content, dpo_remote_path)

    # Train with DPO
    output_dir = f"{VOLUME_PATH}/dpo_models/{model_safe_name}_calibrated"

    print(f"\nStarting DPO training for: {model}")
    result = train_dpo_calibration.remote(
        model_name=model,
        dpo_data_path=dpo_remote_path,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    print(f"\n{'='*80}")
    print("JOB COMPLETE")
    print(f"{'='*80}")
    print(f"Model: {result['model']}")
    print(f"Training examples: {result['train_examples']}")
    print(f"Eval examples: {result['eval_examples']}")
    print(f"\nModel saved to Modal volume at:")
    print(f"  {result['output_dir']}")
    print(f"\nTo download model:")
    print(f"  modal volume get slm-dpo-training {result['output_dir']}")
