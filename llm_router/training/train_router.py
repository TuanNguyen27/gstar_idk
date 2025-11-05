"""
Router Model Training
Implements SFT + DPO training pipeline using QLoRA.
"""

import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DPOTrainer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RouterTrainer:
    """Trains the router model using SFT + DPO."""

    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        output_dir: str = "./router_checkpoints",
    ):
        """
        Initialize router trainer.

        Args:
            model_name: HuggingFace model identifier
            output_dir: Directory to save checkpoints
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # QLoRA configuration for 4-bit quantization
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def train_sft(
        self,
        train_data_path: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_seq_length: int = 1024,
    ) -> str:
        """
        Stage 1: Supervised Fine-Tuning.

        Args:
            train_data_path: Path to SFT training data JSONL
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_seq_length: Maximum sequence length

        Returns:
            Path to saved SFT model checkpoint
        """
        logger.info("Starting SFT training...")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)

        # Load dataset
        dataset = load_dataset("json", data_files=train_data_path, split="train")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "sft"),
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
        )

        # SFT Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            dataset_text_field="prompt",  # Will concatenate prompt + completion
        )

        # Train
        trainer.train()

        # Save model
        sft_output = self.output_dir / "router_base"
        trainer.save_model(str(sft_output))
        logger.info(f"SFT training complete. Model saved to {sft_output}")

        return str(sft_output)

    def train_dpo(
        self,
        sft_model_path: str,
        train_data_path: str,
        num_epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        beta: float = 0.1,
    ) -> str:
        """
        Stage 2: Direct Preference Optimization.

        Args:
            sft_model_path: Path to SFT checkpoint
            train_data_path: Path to DPO training data JSONL
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            beta: DPO beta parameter (temperature)

        Returns:
            Path to saved DPO model checkpoint
        """
        logger.info("Starting DPO training...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load SFT model
        model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            quantization_config=self.bnb_config,
            device_map="auto",
        )

        # Load reference model (frozen)
        model_ref = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            quantization_config=self.bnb_config,
            device_map="auto",
        )

        # Load dataset
        dataset = load_dataset("json", data_files=train_data_path, split="train")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "dpo"),
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
        dpo_trainer.train()

        # Save final model
        dpo_output = self.output_dir / "router_final"
        dpo_trainer.save_model(str(dpo_output))
        logger.info(f"DPO training complete. Model saved to {dpo_output}")

        return str(dpo_output)

    def train_full_pipeline(
        self,
        sft_data_path: str,
        dpo_data_path: str,
        sft_epochs: int = 3,
        dpo_epochs: int = 1,
    ) -> str:
        """
        Run the full SFT + DPO training pipeline.

        Args:
            sft_data_path: Path to SFT training data
            dpo_data_path: Path to DPO training data
            sft_epochs: Number of SFT epochs
            dpo_epochs: Number of DPO epochs

        Returns:
            Path to final trained model
        """
        logger.info("Starting full training pipeline: SFT + DPO")

        # Stage 1: SFT
        sft_model_path = self.train_sft(
            train_data_path=sft_data_path,
            num_epochs=sft_epochs,
        )

        # Stage 2: DPO
        final_model_path = self.train_dpo(
            sft_model_path=sft_model_path,
            train_data_path=dpo_data_path,
            num_epochs=dpo_epochs,
        )

        logger.info(f"Full training pipeline complete. Final model: {final_model_path}")
        return final_model_path
