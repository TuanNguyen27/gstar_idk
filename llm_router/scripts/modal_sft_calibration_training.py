"""
Modal script for standard SFT calibration training.
Simpler than DPO - just direct supervision on (question, answer+confidence) pairs.
"""

import modal

app = modal.App("slm-sft-calibration")

# Training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",
        "torch==2.1.0",
        "transformers==4.42.0",
        "peft==0.11.0",
        "datasets==2.15.0",
        "accelerate==0.30.0",
        "bitsandbytes==0.43.0",
        "trl==0.8.6",
        "scipy",
    )
)

volume = modal.Volume.from_name("slm-dpo-training", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(
    image=training_image,
    gpu="A100-40GB",
    timeout=14400,  # 4 hours
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_sft_calibration(
    model_name: str,
    sft_data_path: str,
    output_dir: str,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    use_wandb: bool = False,
):
    """
    Train model using standard SFT for calibration.

    Args:
        model_name: Base model to fine-tune
        sft_data_path: Path to SFT data (on Modal volume)
        output_dir: Where to save trained model
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        use_wandb: Whether to use wandb logging
    """
    import json
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from pathlib import Path

    print(f"\n{'='*80}")
    print(f"SFT Calibration Training: {model_name}")
    print(f"{'='*80}\n")

    if not use_wandb:
        print("Wandb not available, skipping logging")

    # Load SFT data
    print(f"Loading SFT data from {sft_data_path}")
    sft_examples = []
    with open(sft_data_path, 'r') as f:
        for line in f:
            sft_examples.append(json.loads(line))

    print(f"Loaded {len(sft_examples)} SFT examples")

    # Split into train/eval
    split_idx = int(len(sft_examples) * 0.9)
    train_examples = sft_examples[:split_idx]
    eval_examples = sft_examples[split_idx:]

    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")

    # Create dataset
    def format_examples(examples):
        return {
            "input": [ex["input"] for ex in examples],
            "output": [ex["output"] for ex in examples],
        }

    train_data = format_examples(train_examples)
    eval_data = format_examples(eval_examples)

    dataset = {
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(eval_data),
    }

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_function(examples):
        # Format: input + output
        prompts = [
            f"{inp}\n\n{out}"
            for inp, out in zip(examples["input"], examples["output"])
        ]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = {
        "train": dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=["input", "output"],
        ),
        "test": dataset["test"].map(
            tokenize_function,
            batched=True,
            remove_columns=["input", "output"],
        ),
    }

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    print("Applying LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
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
        remove_unused_columns=True,
    )

    # Initialize trainer
    print("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    # Train
    print("\nStarting SFT calibration training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Verify files were saved
    saved_files = list(Path(output_dir).glob("*"))
    print(f"Saved {len(saved_files)} files to {output_dir}:")
    for f in saved_files[:10]:  # Show first 10 files
        print(f"  - {f.name}")

    # Commit volume
    volume.commit()
    print(f"Volume committed successfully")

    print(f"\n{'='*80}")
    print("SFT CALIBRATION TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model saved to: {output_dir}")

    return {
        "model": model_name,
        "output_dir": output_dir,
        "train_examples": len(tokenized_dataset["train"]),
        "eval_examples": len(tokenized_dataset["test"]),
    }


@app.function(volumes={VOLUME_PATH: volume})
def upload_sft_data(content: str, remote_path: str):
    """Upload SFT data to Modal volume."""
    from pathlib import Path

    # Create parent directory if it doesn't exist
    remote_file = Path(remote_path)
    remote_file.parent.mkdir(parents=True, exist_ok=True)

    with open(remote_path, 'w') as f:
        f.write(content)
    volume.commit()
    print(f"SFT data uploaded to {remote_path}")


@app.local_entrypoint()
def main(
    model: str = "google/gemma-2-2b-it",
    sft_data: str = "data/slm_baseline/google_gemma-2-2b-it_sft_calibration.jsonl",
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
):
    """
    Main entrypoint for SFT calibration training.

    Usage:
        modal run scripts/modal_sft_calibration_training.py \
            --model google/gemma-2-2b-it \
            --sft-data data/slm_baseline/google_gemma-2-2b-it_sft_calibration.jsonl \
            --learning-rate 1e-5 \
            --num-epochs 3
    """
    from pathlib import Path

    print(f"{'='*80}")
    print("SFT CALIBRATION TRAINING")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"SFT data: {sft_data}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print()

    # Upload SFT data to Modal
    print(f"Uploading SFT data: {sft_data}")
    with open(sft_data, 'r') as f:
        sft_content = f.read()

    sft_remote_path = f"{VOLUME_PATH}/sft_data/{Path(sft_data).name}"
    upload_sft_data.remote(sft_content, sft_remote_path)

    # Prepare output directory
    model_safe_name = model.replace("/", "_")
    output_dir = f"{VOLUME_PATH}/sft_calibrated_models/{model_safe_name}_calibrated"

    print(f"\nStarting SFT calibration training for: {model}")
    result = train_sft_calibration.remote(
        model_name=model,
        sft_data_path=sft_remote_path,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model saved to:")
    print(f"  {result['output_dir']}")
    print(f"\nTo download the model:")
    print(f"  modal volume get slm-dpo-training {result['output_dir']}")
