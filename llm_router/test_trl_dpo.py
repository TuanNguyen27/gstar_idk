#!/usr/bin/env python3
"""
Quick test to verify trl==0.8.6 supports DPOTrainer with beta parameter.
"""

def test_trl_version():
    """Test that trl 0.8.6 has DPOTrainer with beta support."""
    try:
        import trl
        print(f"✅ trl version: {trl.__version__}")
    except ImportError:
        print("❌ trl not installed")
        return False

    # Test DPOTrainer import
    try:
        from trl import DPOTrainer
        print(f"✅ DPOTrainer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DPOTrainer: {e}")
        return False

    # Test DPOTrainer signature
    import inspect
    sig = inspect.signature(DPOTrainer.__init__)
    params = list(sig.parameters.keys())

    print(f"\n📋 DPOTrainer.__init__ parameters:")
    for param in params[:10]:  # Show first 10 params
        print(f"  - {param}")
    if len(params) > 10:
        print(f"  ... and {len(params) - 10} more")

    # Check if beta parameter exists
    if 'beta' in params:
        print(f"\n✅ 'beta' parameter found in DPOTrainer")
        return True
    else:
        print(f"\n❌ 'beta' parameter NOT found in DPOTrainer")
        print(f"   Available params: {params}")
        return False

def test_training_args_compatibility():
    """Test that TrainingArguments works with trl 0.8.6."""
    try:
        from transformers import TrainingArguments
        print(f"\n✅ TrainingArguments imported successfully")

        # Test creating a minimal TrainingArguments
        args = TrainingArguments(
            output_dir="/tmp/test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
        )
        print(f"✅ TrainingArguments instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create TrainingArguments: {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("TRL 0.8.6 DPOTrainer Verification")
    print("="*80)
    print()

    result1 = test_trl_version()
    result2 = test_training_args_compatibility()

    print()
    print("="*80)
    if result1 and result2:
        print("✅ ALL TESTS PASSED - trl 0.8.6 is compatible!")
    else:
        print("❌ TESTS FAILED - there may be compatibility issues")
    print("="*80)
