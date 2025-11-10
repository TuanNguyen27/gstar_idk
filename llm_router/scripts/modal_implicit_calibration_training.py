"""
Modal script for implicit calibration training using logprobs.
Trains models to be better calibrated by penalizing miscalibration in the loss function.
Does NOT require changing model output format.
"""

import modal
import json
from pathlib import Path

app = modal.App("slm-implicit-calibration")

# Training image
calibration_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy<2",  # Pin numpy to <2 for compatibility
        "scipy",  # Required by bitsandbytes
        "torch==2.1.0",
        "transformers==4.42.0",  # Newer version for Gemma support
        "peft==0.11.0",  # For LoRA
        "datasets==2.15.0",
        "accelerate==0.30.0",
        "bitsandbytes==0.43.0",  # For quantization
        "wandb",
        "tqdm",
    )
)

volume = modal.Volume.from_name("slm-implicit-calibration", create_if_missing=True)
VOLUME_PATH = "/vol"


@app.function(
    image=calibration_image,
    gpu="A100-40GB",
    timeout=3600 * 4,
    volumes={VOLUME_PATH: volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_implicit_calibration(
    model_name: str,
    judged_data_path: str,
    output_dir: str,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    calibration_weight: float = 0.1,
):
    """
    Train SLM with implicit calibration using custom loss.

    Key idea: Penalize miscalibration in the training loss without changing output format.

    Args:
        model_name: HuggingFace model ID
        judged_data_path: Path to judged results (with correctness labels)
        output_dir: Directory to save trained model
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        calibration_weight: Weight for calibration loss component (0.1 = 10%)
    """
    import os
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    import wandb
    import numpy as np

    print(f"\n{'='*80}")
    print(f"Implicit Calibration Training: {model_name}")
    print(f"{'='*80}\n")

    # Initialize wandb (optional)
    try:
        wandb.init(
            project="slm-implicit-calibration",
            name=f"{model_name.replace('/', '_')}_implicit",
            config={
                "model": model_name,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "calibration_weight": calibration_weight,
            }
        )
        use_wandb = True
    except:
        print("Wandb not available, skipping logging")
        use_wandb = False

    # Load judged data
    print(f"Loading judged data from {judged_data_path}")
    judged_examples = []
    with open(judged_data_path, 'r') as f:
        for line in f:
            judged_examples.append(json.loads(line))

    print(f"Loaded {len(judged_examples)} judged examples")

    # Prepare dataset
    dataset_dict = {
        "question": [ex["problem"] for ex in judged_examples],
        "answer": [ex["model_answer"] for ex in judged_examples],
        "is_correct": [int(ex["is_correct"]) for ex in judged_examples],
    }
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Train examples: {len(dataset['train'])}")
    print(f"Eval examples: {len(dataset['test'])}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_function(examples):
        prompts = [
            f"Question: {q}\n\nAnswer: {a}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        tokenized["is_correct"] = examples["is_correct"]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["question", "answer"],
    )

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
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

    # Custom Trainer with calibration loss
    class CalibrationTrainer(Trainer):
        def __init__(self, *args, calibration_weight=0.1, **kwargs):
            super().__init__(*args, **kwargs)
            self.calibration_weight = calibration_weight

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            Custom loss = LM loss + calibration penalty.

            Calibration penalty = MSE(avg_prob, is_correct)
            """
            # Extract correctness labels
            is_correct = inputs.pop("is_correct").float()

            # Standard LM loss (pass labels to get loss automatically)
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)

            # Extract loss robustly
            if isinstance(outputs, dict):
                lm_loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                lm_loss = outputs.loss
            else:
                lm_loss = outputs[0]

            # Compute calibration loss
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            labels = inputs["input_ids"]

            # Get probabilities for generated tokens
            # Shape: [batch_size, seq_len, vocab_size]
            probs = F.softmax(logits, dim=-1)

            # Get probability of actual tokens
            # Shape: [batch_size, seq_len]
            token_probs = torch.gather(
                probs, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)

            # Mask padding tokens
            attention_mask = inputs["attention_mask"]
            token_probs = token_probs * attention_mask

            # Compute average probability per example (geometric mean)
            log_probs = torch.log(token_probs + 1e-10)
            avg_log_prob = (log_probs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            avg_prob = torch.exp(avg_log_prob)

            # Calibration loss: penalize deviation from correctness
            # If correct → want high confidence (avg_prob → 1)
            # If incorrect → want low confidence (avg_prob → 0)
            calibration_loss = F.mse_loss(avg_prob, is_correct)

            # Combined loss
            total_loss = lm_loss + self.calibration_weight * calibration_loss

            # Log metrics
            self.log({
                "lm_loss": lm_loss.item(),
                "calibration_loss": calibration_loss.item(),
                "avg_confidence": avg_prob.mean().item(),
            })

            return (total_loss, outputs) if return_outputs else total_loss

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
        remove_unused_columns=False,
        label_names=["is_correct"],  # Prevent is_correct from being passed to model during eval
    )

    # Initialize custom trainer
    print("\nInitializing Calibration Trainer")
    trainer = CalibrationTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        calibration_weight=calibration_weight,
    )

    # Train
    print("\nStarting calibration training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}")

    # Create output directory if it doesn't exist
    from pathlib import Path
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
    print("IMPLICIT CALIBRATION TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model saved to: {output_dir}")

    return {
        "model": model_name,
        "output_dir": output_dir,
        "train_examples": len(tokenized_dataset["train"]),
        "eval_examples": len(tokenized_dataset["test"]),
    }


@app.function(volumes={VOLUME_PATH: volume})
def upload_judged_data(content: str, remote_path: str):
    """Upload judged data to Modal volume."""
    from pathlib import Path

    # Create parent directory if it doesn't exist
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
    calibration_weight: float = 0.1,
):
    """
    Main entrypoint for implicit calibration training.

    Usage:
        modal run scripts/modal_implicit_calibration_training.py \\
            --model google/gemma-2-2b-it \\
            --judged-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \\
            --calibration-weight 0.1

    Key difference from DPO approach:
        - Does NOT change model output format
        - Trains on logprobs directly
        - Adds calibration penalty to standard LM loss
    """
    print(f"{'='*80}")
    print("IMPLICIT CALIBRATION TRAINING")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Judged data: {judged_data}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Calibration weight: {calibration_weight}")
    print()

    # Upload judged data to Modal
    print(f"Uploading judged data: {judged_data}")
    with open(judged_data, 'r') as f:
        judged_content = f.read()

    model_safe_name = model.replace('/', '_')
    judged_remote_path = f"{VOLUME_PATH}/judged_data/{model_safe_name}_judged.jsonl"
    upload_judged_data.remote(judged_content, judged_remote_path)

    # Train with implicit calibration
    output_dir = f"{VOLUME_PATH}/implicit_calibrated_models/{model_safe_name}_calibrated"

    print(f"\nStarting implicit calibration training for: {model}")
    result = train_implicit_calibration.remote(
        model_name=model,
        judged_data_path=judged_remote_path,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        calibration_weight=calibration_weight,
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
    print(f"  modal volume get slm-implicit-calibration {result['output_dir']}")
