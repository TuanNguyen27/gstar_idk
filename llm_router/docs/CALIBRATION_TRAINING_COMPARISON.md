# Calibration Training Approaches: Comparison

## Overview

We've implemented **two approaches** to improve SLM calibration:

1. **Option A: Implicit Calibration** (logprobs-based)
2. **Option B: Explicit Confidence DPO** (text generation-based)

Both approaches aim to reduce RMS calibration error, but use different mechanisms.

---

## Option A: Implicit Calibration

**Script**: `scripts/modal_implicit_calibration_training.py`

### How It Works

```python
# Custom loss function
loss = lm_loss + λ * calibration_penalty

# Where calibration_penalty = MSE(avg_token_prob, is_correct)
```

**Training objective**:
- If answer is correct → maximize average token probability
- If answer is incorrect → minimize average token probability

### Pros ✅

1. **No format change** - model output stays the same
2. **Theoretically cleaner** - directly optimizes internal representations
3. **Backward compatible** - can use with existing inference pipeline
4. **Efficient** - single training pass, no need for preference pairs

### Cons ❌

1. **Custom training loop** - requires implementing custom Trainer
2. **Indirect supervision** - only penalizes average confidence, not per-token
3. **May affect answer quality** - training changes token selection
4. **Harder to debug** - calibration improvements are implicit

### Use Cases

- When you want to keep existing inference pipeline
- When you care about minimizing changes to model behavior
- When you want fast iteration (no need to generate preference pairs)

---

## Option B: Explicit Confidence DPO

**Script**: `scripts/modal_dpo_training.py`

### How It Works

```python
# DPO prefers well-calibrated outputs
chosen = "Answer: Paris\nConfidence: 0.90"    # Well-calibrated
rejected = "Answer: Paris\nConfidence: 0.50"  # Poorly-calibrated

# Standard DPO loss
loss = -log σ(β * (r_chosen - r_rejected))
```

**Training objective**:
- Model learns to explicitly generate calibrated confidence scores

### Pros ✅

1. **Uses standard DPO** - no custom code, proven libraries (TRL)
2. **Explicit outputs** - confidence is visible and interpretable
3. **Clean separation** - answer quality vs confidence calibration
4. **Easy to debug** - can inspect confidence scores directly
5. **Human-interpretable** - model states its confidence

### Cons ❌

1. **Format change required** - must request confidence in prompt
2. **Longer generation** - extra tokens for confidence output
3. **Need preference pairs** - requires generating training data first
4. **Synthetic preferences** - uses constructed pairs, not real comparisons

### Use Cases

- When you want interpretable confidence scores
- When you're okay with changing inference format
- When you want to use proven DPO libraries
- When you want clear separation between answer and confidence

---

## Comparison Table

| Aspect | Implicit (A) | Explicit DPO (B) |
|--------|-------------|------------------|
| **Output format** | Unchanged | "Answer: X\nConfidence: Y" |
| **Training complexity** | Custom loss | Standard DPO |
| **Inference** | Extract logprobs | Parse text |
| **Interpretability** | Hidden | Visible |
| **Training data** | Judged results only | Preference pairs needed |
| **Implementation** | Custom Trainer | TRL library |
| **Backward compatible** | ✅ Yes | ❌ No (new format) |
| **Development time** | High (custom code) | Low (standard tools) |
| **Debuggability** | Hard | Easy |

---

## Experimental Setup

### Baseline Measurement

1. **Inference**: Run vLLM on 1000 SimpleQA examples
2. **Judge**: Use Gemini Pro to label correctness
3. **Measure**: Calculate RMS calibration error with HLE's metric

### Training

**Both approaches**:
- LoRA training (r=16, alpha=32)
- 3 epochs
- Learning rate: 1e-5
- Batch size: 4 (with gradient accumulation)
- GPU: A100-40GB

### Evaluation

1. **Re-run inference** on same 1000 examples
2. **Extract confidence**:
   - Implicit: logprobs (same as baseline)
   - Explicit: parse from text output
3. **Calculate RMS calibration error** (compare to baseline)

---

## Expected Results

### Hypothesis

**Implicit calibration**:
- Expected improvement: 10-20% reduction in RMS error
- May slightly affect answer accuracy (±2%)
- Confidence extraction: same method as baseline

**Explicit DPO**:
- Expected improvement: 20-40% reduction in RMS error
- Answer accuracy: unchanged (only confidence changes)
- Confidence extraction: parse from generated text

### Success Criteria

Both approaches succeed if:
- RMS calibration error decreases by ≥10%
- Answer accuracy doesn't degrade by >5%
- Confidence scores span reasonable range (not collapsed)

---

## Recommendation

**Start with Option B (Explicit DPO)** because:
1. ✅ Easier to implement (standard libraries)
2. ✅ Clearer to debug (visible confidence)
3. ✅ Proven approach (DPO is well-studied)
4. ✅ Clean separation (answer vs confidence)

**Try Option A (Implicit) if**:
- Explicit approach doesn't work well
- You need backward compatibility
- You want to avoid format changes

---

## Next Steps

1. ✅ Wait for judging to complete (~40 min remaining)
2. Generate DPO preference pairs (Option B)
3. Train both approaches in parallel
4. Compare results
5. Iterate on best-performing approach

---

## Implementation Status

| Task | Status | Script |
|------|--------|--------|
| Baseline inference | ✅ Complete | `scripts/modal_slm_baseline.py` |
| Judging | 🔄 In progress (17%) | `evaluation/judge_slm_answers.py` |
| DPO data generation | ✅ Ready | `training/generate_dpo_calibration_data.py` |
| Explicit DPO training | ✅ Ready | `scripts/modal_dpo_training.py` |
| Implicit training | ✅ Ready | `scripts/modal_implicit_calibration_training.py` |
| Post-training evaluation | ⏳ TODO | TBD |
| Comparison analysis | ⏳ TODO | TBD |

---

## Cost Estimate

**Per model (Gemma-2B or Qwen-1.5B)**:
- DPO data generation: $0 (local)
- Training (3 epochs, A100): ~$5-10 (1-2 hours)
- Post-training inference: ~$2 (vLLM on Modal)
- **Total per approach**: ~$7-12
- **Both approaches**: ~$15-25

**For 2 models × 2 approaches**: ~$30-50 total
