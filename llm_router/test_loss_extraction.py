#!/usr/bin/env python3
"""
Local test to verify loss extraction logic from model outputs.
Tests different transformers versions and output formats.
"""

def test_loss_extraction():
    """Test loss extraction from different output formats."""
    import torch

    print("="*80)
    print("Testing Loss Extraction Logic")
    print("="*80)
    print()

    # Test Case 1: Standard ModelOutput with loss attribute
    print("Test 1: ModelOutput with .loss attribute")
    from dataclasses import dataclass

    @dataclass
    class ModelOutput1:
        loss: torch.Tensor
        logits: torch.Tensor

    outputs = ModelOutput1(
        loss=torch.tensor(2.5),
        logits=torch.randn(2, 10, 100)
    )

    # Test our extraction logic
    if isinstance(outputs, dict):
        lm_loss = outputs['loss']
    elif hasattr(outputs, 'loss'):
        lm_loss = outputs.loss
    else:
        lm_loss = outputs[0]

    print(f"  outputs type: {type(outputs)}")
    print(f"  hasattr(outputs, 'loss'): {hasattr(outputs, 'loss')}")
    print(f"  extracted loss: {lm_loss}")
    print(f"  loss is tensor: {isinstance(lm_loss, torch.Tensor)}")
    print(f"  ✅ PASS\n")

    # Test Case 2: Dict output with 'loss' key
    print("Test 2: Dict with 'loss' key")
    outputs = {
        'loss': torch.tensor(2.5),
        'logits': torch.randn(2, 10, 100)
    }

    if isinstance(outputs, dict):
        lm_loss = outputs['loss']
    elif hasattr(outputs, 'loss'):
        lm_loss = outputs.loss
    else:
        lm_loss = outputs[0]

    print(f"  outputs type: {type(outputs)}")
    print(f"  isinstance(outputs, dict): {isinstance(outputs, dict)}")
    print(f"  extracted loss: {lm_loss}")
    print(f"  loss is tensor: {isinstance(lm_loss, torch.Tensor)}")
    print(f"  ✅ PASS\n")

    # Test Case 3: Tuple output (legacy format)
    print("Test 3: Tuple output (loss, logits)")
    outputs = (torch.tensor(2.5), torch.randn(2, 10, 100))

    if isinstance(outputs, dict):
        lm_loss = outputs['loss']
    elif hasattr(outputs, 'loss'):
        lm_loss = outputs.loss
    else:
        lm_loss = outputs[0]

    print(f"  outputs type: {type(outputs)}")
    print(f"  extracted loss: {lm_loss}")
    print(f"  loss is tensor: {isinstance(lm_loss, torch.Tensor)}")
    print(f"  ✅ PASS\n")

    # Test Case 4: Nested dict (problematic case)
    print("Test 4: Nested dict with loss as dict (PROBLEMATIC)")
    outputs = {
        'loss': {'value': torch.tensor(2.5)},  # This is BAD!
        'logits': torch.randn(2, 10, 100)
    }

    if isinstance(outputs, dict):
        lm_loss = outputs['loss']
    elif hasattr(outputs, 'loss'):
        lm_loss = outputs.loss
    else:
        lm_loss = outputs[0]

    print(f"  outputs type: {type(outputs)}")
    print(f"  extracted loss: {lm_loss}")
    print(f"  loss type: {type(lm_loss)}")
    print(f"  loss is tensor: {isinstance(lm_loss, torch.Tensor)}")
    print(f"  ❌ FAIL - loss is dict, not tensor!\n")

    # Test Case 5: Test actual model call (if transformers available)
    try:
        print("Test 5: Real model call with transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import warnings
        warnings.filterwarnings('ignore')

        # Load tiny model for testing
        print("  Loading tiny model (gpt2)...")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Create test input
        text = "Hello world"
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()

        # Forward pass with labels
        print("  Running forward pass with labels...")
        outputs = model(**inputs, labels=labels)

        # Test extraction
        if isinstance(outputs, dict):
            lm_loss = outputs['loss']
        elif hasattr(outputs, 'loss'):
            lm_loss = outputs.loss
        else:
            lm_loss = outputs[0]

        print(f"  outputs type: {type(outputs)}")
        print(f"  hasattr(outputs, 'loss'): {hasattr(outputs, 'loss')}")
        print(f"  isinstance(outputs, dict): {isinstance(outputs, dict)}")
        print(f"  extracted loss: {lm_loss}")
        print(f"  loss is tensor: {isinstance(lm_loss, torch.Tensor)}")

        # Try arithmetic
        test_loss = lm_loss + torch.tensor(1.0)
        print(f"  lm_loss + 1.0 = {test_loss}")
        print(f"  ✅ PASS - can do arithmetic!\n")

    except ImportError:
        print("  ⚠️  Transformers not installed, skipping real model test\n")
    except Exception as e:
        print(f"  ❌ FAIL: {e}\n")

    print("="*80)
    print("Summary:")
    print("  Our logic handles:")
    print("    ✅ dict outputs")
    print("    ✅ ModelOutput with .loss")
    print("    ✅ tuple outputs")
    print("  Potential issue:")
    print("    ❌ Nested dict where outputs['loss'] is itself a dict")
    print("="*80)

if __name__ == "__main__":
    test_loss_extraction()
