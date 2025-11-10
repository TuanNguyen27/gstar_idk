#!/usr/bin/env python3
"""
Local test to simulate implicit calibration trainer's compute_loss method.
Performs actual forward/backward pass and inspects output types.
"""

def test_implicit_calibration_loss():
    """Test implicit calibration compute_loss with real model."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from datasets import Dataset
    import warnings
    warnings.filterwarnings('ignore')

    print("="*80)
    print("Testing Implicit Calibration Trainer")
    print("="*80)
    print()

    # Load tiny model for testing
    print("Loading tiny model (gpt2)...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dummy dataset
    print("Creating dummy dataset...")
    dataset_dict = {
        "input_ids": [
            tokenizer("Question: What is 2+2? Answer: 4").input_ids[:20],
            tokenizer("Question: What is 3+3? Answer: 6").input_ids[:20],
        ],
        "attention_mask": [
            [1] * 20,
            [1] * 20,
        ],
        "is_correct": [1, 0],  # Binary labels
    }
    dataset = Dataset.from_dict(dataset_dict)

    # Custom Trainer with calibration loss
    class CalibrationTrainer(Trainer):
        def __init__(self, *args, calibration_weight=0.1, **kwargs):
            super().__init__(*args, **kwargs)
            self.calibration_weight = calibration_weight

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            Custom loss = LM loss + calibration penalty.
            """
            # Extract correctness labels
            is_correct = inputs.pop("is_correct").float()

            # Standard LM loss
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)

            print(f"\n{'='*60}")
            print(f"Forward pass output inspection:")
            print(f"  Type: {type(outputs)}")
            print(f"  isinstance(outputs, dict): {isinstance(outputs, dict)}")
            print(f"  hasattr(outputs, 'loss'): {hasattr(outputs, 'loss')}")

            if isinstance(outputs, dict):
                print(f"  Dict keys: {outputs.keys()}")
                print(f"  outputs['loss'] type: {type(outputs['loss'])}")
                print(f"  outputs['loss'] value: {outputs['loss']}")
            elif hasattr(outputs, 'loss'):
                print(f"  outputs.loss type: {type(outputs.loss)}")
                print(f"  outputs.loss value: {outputs.loss}")

            # Extract loss robustly
            if isinstance(outputs, dict):
                lm_loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                lm_loss = outputs.loss
            else:
                lm_loss = outputs[0]

            print(f"\n  Extracted lm_loss type: {type(lm_loss)}")
            print(f"  Extracted lm_loss value: {lm_loss}")
            print(f"  Is tensor: {isinstance(lm_loss, torch.Tensor)}")

            # Compute calibration loss
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            probs = F.softmax(logits, dim=-1)
            token_probs = torch.gather(
                probs, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)

            # Mask padding
            attention_mask = inputs["attention_mask"]
            token_probs = token_probs * attention_mask

            # Geometric mean
            log_probs = torch.log(token_probs + 1e-10)
            avg_log_prob = (log_probs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            avg_prob = torch.exp(avg_log_prob)

            # Calibration loss
            calibration_loss = F.mse_loss(avg_prob, is_correct)

            print(f"\n  Calibration loss type: {type(calibration_loss)}")
            print(f"  Calibration loss value: {calibration_loss}")

            # Combined loss
            try:
                total_loss = lm_loss + self.calibration_weight * calibration_loss
                print(f"\n  ✅ SUCCESS: total_loss = {total_loss}")
                print(f"  total_loss type: {type(total_loss)}")
            except TypeError as e:
                print(f"\n  ❌ FAIL: Cannot compute total_loss")
                print(f"  Error: {e}")
                print(f"  lm_loss type: {type(lm_loss)}")
                print(f"  calibration_loss type: {type(calibration_loss)}")
                raise

            print(f"{'='*60}\n")

            return (total_loss, outputs) if return_outputs else total_loss

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/test_calibration",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb
    )

    # Initialize trainer
    print("Initializing Calibration Trainer...")
    trainer = CalibrationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        calibration_weight=0.1,
    )

    # Run one training step
    print("\nRunning one training step...\n")
    try:
        trainer.train()
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)

if __name__ == "__main__":
    test_implicit_calibration_loss()
