"""
Modal-based Router Training
Distributed SFT + DPO training using Modal's GPU infrastructure.
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("llm-router-training")

# Define the training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0",
        "scipy",
        "sentencepiece",
        "protobuf",
    )
)

# Create persistent volume for model checkpoints
volume = modal.Volume.from_name("llm-router-models", create_if_missing=True)

@app.function(
    image=training_image,
    gpu="A100",  # or "A10G" for smaller models
    timeout=3600 * 4,  # 4 hours
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For model downloads
)
def train_sft_stage(
    model_name: str,
    train_data_jsonl: bytes,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
) -> str:
    """
    Stage 1: Supervised Fine-Tuning on Modal.

    Args:
        model_name: HuggingFace model identifier
        train_data_jsonl: Training data as JSONL bytes
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate

    Returns:
        Path to saved SFT model on volume
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset
    import tempfile
    import os

    print(f"Starting SFT training for {model_name}")

    # Write training data to temp file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl', delete=False) as f:
        f.write(train_data_jsonl)
        train_data_path = f.name

    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    print(f"Trainable parameters: {model.print_trainable_parameters()}")

    # Load dataset
    dataset = load_dataset("json", data_files=train_data_path, split="train")
    print(f"Loaded {len(dataset)} training examples")

    # Training arguments
    output_dir = "/vol/sft_checkpoints"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=1024,
        dataset_text_field="prompt",
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    model_output = "/vol/router_base"
    trainer.save_model(model_output)
    print(f"SFT training complete. Model saved to {model_output}")

    # Cleanup
    os.unlink(train_data_path)
    volume.commit()

    return model_output

@app.function(
    image=training_image,
    gpu="A100",
    timeout=3600 * 3,  # 3 hours
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_dpo_stage(
    base_model_name: str,
    sft_model_volume_path: str,
    train_data_jsonl: bytes,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
) -> str:
    """
    Stage 2: Direct Preference Optimization on Modal.

    Args:
        base_model_name: Original HuggingFace model identifier
        sft_model_volume_path: Path to SFT checkpoint on volume
        train_data_jsonl: Training data as JSONL bytes
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        beta: DPO beta parameter

    Returns:
        Path to final model on volume
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from trl import DPOTrainer
    from datasets import load_dataset
    import tempfile
    import os

    print(f"Starting DPO training from {sft_model_volume_path}")

    # Write training data to temp file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl', delete=False) as f:
        f.write(train_data_jsonl)
        train_data_path = f.name

    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model
    print("Loading SFT model...")
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_volume_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load reference model (frozen)
    model_ref = AutoModelForCausalLM.from_pretrained(
        sft_model_volume_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load dataset
    dataset = load_dataset("json", data_files=train_data_path, split="train")
    print(f"Loaded {len(dataset)} training examples")

    # Training arguments
    output_dir = "/vol/dpo_checkpoints"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=beta,
    )

    # Train
    print("Starting DPO training...")
    dpo_trainer.train()

    # Save final model
    model_output = "/vol/router_final"
    dpo_trainer.save_model(model_output)
    print(f"DPO training complete. Final model saved to {model_output}")

    # Cleanup
    os.unlink(train_data_path)
    volume.commit()

    return model_output

@app.local_entrypoint()
def train_full_pipeline(
    model_name: str = "google/gemma-2b",
    sft_data_path: str = "./data/sft_train.jsonl",
    dpo_data_path: str = "./data/dpo_train.jsonl",
    sft_epochs: int = 3,
    dpo_epochs: int = 1,
):
    """
    Run the full SFT + DPO training pipeline on Modal.

    Args:
        model_name: HuggingFace model identifier
        sft_data_path: Path to SFT training data
        dpo_data_path: Path to DPO training data
        sft_epochs: Number of SFT epochs
        dpo_epochs: Number of DPO epochs
    """
    print("=== LLM Router Training Pipeline ===")
    print(f"Model: {model_name}")
    print(f"SFT epochs: {sft_epochs}, DPO epochs: {dpo_epochs}")

    # Read training data
    with open(sft_data_path, 'rb') as f:
        sft_data = f.read()

    with open(dpo_data_path, 'rb') as f:
        dpo_data = f.read()

    # Stage 1: SFT
    print("\n=== Stage 1: Supervised Fine-Tuning ===")
    sft_model_path = train_sft_stage.remote(
        model_name=model_name,
        train_data_jsonl=sft_data,
        num_epochs=sft_epochs,
    )
    print(f"✓ SFT complete: {sft_model_path}")

    # Stage 2: DPO
    print("\n=== Stage 2: Direct Preference Optimization ===")
    final_model_path = train_dpo_stage.remote(
        base_model_name=model_name,
        sft_model_volume_path=sft_model_path,
        train_data_jsonl=dpo_data,
        num_epochs=dpo_epochs,
    )
    print(f"✓ DPO complete: {final_model_path}")

    print("\n=== Training Complete ===")
    print(f"Final model saved to Modal volume: {final_model_path}")
    print("Download with: modal volume get llm-router-models router_final ./local_model")
