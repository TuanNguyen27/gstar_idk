"""
Modal script for training performance-based continuous confidence SFT model.
Uses k-NN derived continuous confidence labels for improved calibration.
"""

import modal
import json
from pathlib import Path

app = modal.App("slm-performance-sft-training")

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
        "trl==0.8.6",  # Use 0.8.6 for stability
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
def train_performance_sft_model(
    model_name: str,
    sft_data_path: str,
    output_path: str,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    """
    Train SFT model with performance-based continuous confidence labels.

    Args:
        model_name: Base model to fine-tune
        sft_data_path: Path to SFT training data (on Modal volume)
        output_path: Where to save trained model (on Modal volume)
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset
    import numpy as np

    print(f"\n{'='*80}")
    print(f"Training Performance-Based SFT Model")
    print(f"{'='*80}")
    print(f"Base model: {model_name}")
    print(f"Training data: {sft_data_path}")
    print(f"Output: {output_path}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*80}\n")

    # Load training data
    print(f"Loading training data from {sft_data_path}")
    training_examples = []
    with open(sft_data_path, 'r') as f:
        for line in f:
            training_examples.append(json.loads(line))

    print(f"Loaded {len(training_examples)} training examples")

    # Statistics
    confidences = [float(ex["output"].split("Confidence: ")[1]) for ex in training_examples]
    print(f"\nTraining data statistics:")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Std confidence: {np.std(confidences):.3f}")
    print(f"  Min confidence: {np.min(confidences):.3f}")
    print(f"  Max confidence: {np.max(confidences):.3f}")

    # Create dataset
    def format_prompt(example):
        """Format example as single text for SFT."""
        return {
            "text": f"{example['input']}\n{example['output']}"
        }

    dataset = Dataset.from_list(training_examples)
    dataset = dataset.map(format_prompt)

    print(f"\nDataset prepared with {len(dataset)} examples")

    # Load tokenizer and model
    print(f"\nLoading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for decoder-only models

    print(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,  # 4-bit quantization
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/training_output",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    trainer.train()

    # Save model
    print(f"\nSaving model to {output_path}")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Commit volume
    volume.commit()

    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"Model saved to: {output_path}")
    print(f"{'='*80}\n")

    return {"status": "success", "output_path": output_path}


@app.function(volumes={VOLUME_PATH: volume})
def upload_training_data(content: str, remote_path: str):
    """Upload training data to Modal volume."""
    from pathlib import Path

    # Create parent directory
    remote_file = Path(remote_path)
    remote_file.parent.mkdir(parents=True, exist_ok=True)

    with open(remote_path, 'w') as f:
        f.write(content)
    volume.commit()
    print(f"Training data uploaded to {remote_path}")


@app.local_entrypoint()
def main(
    model: str = "google/gemma-2-2b-it",
    sft_data: str = "data/slm_baseline/google_gemma-2-2b-it_performance_sft.jsonl",
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
):
    """
    Train performance-based SFT model with continuous confidence labels.

    Usage:
        modal run scripts/modal_performance_sft_training.py \\
            --model google/gemma-2-2b-it \\
            --sft-data data/slm_baseline/google_gemma-2-2b-it_performance_sft.jsonl \\
            --learning-rate 1e-5 \\
            --num-epochs 3
    """
    from pathlib import Path

    print(f"{'='*80}")
    print("PERFORMANCE-BASED SFT TRAINING")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Training data: {sft_data}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*80}\n")

    # Upload training data to Modal
    print(f"Uploading training data: {sft_data}")
    with open(sft_data, 'r') as f:
        training_content = f.read()

    model_slug = model.replace("/", "_")
    remote_path = f"{VOLUME_PATH}/sft_training_data/{model_slug}_performance_sft.jsonl"
    upload_training_data.remote(training_content, remote_path)

    # Train model
    output_path = f"{VOLUME_PATH}/performance_sft_models/{model_slug}_performance_calibrated"

    print(f"\nStarting training job...")
    result = train_performance_sft_model.remote(
        model_name=model,
        sft_data_path=remote_path,
        output_path=output_path,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Status: {result['status']}")
    print(f"Model saved to: {result['output_path']}")
    print(f"\nTo download model:")
    print(f"  modal volume get slm-dpo-training {result['output_path']}")
