# SLM Calibration Training: Data Generation Rationale & Baseline Analysis

## Executive Summary

This document explains the **rationale behind our data generation approach** for improving Small Language Model (SLM) calibration, covering both **baseline measurement** and **two training methods** (DPO and Implicit Calibration). We detail the motivation, methodology, and expected outcomes for each approach.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Baseline Measurement](#baseline-measurement)
3. [Data Generation Philosophy](#data-generation-philosophy)
4. [Approach A: Explicit Confidence DPO](#approach-a-explicit-confidence-dpo)
5. [Approach B: Implicit Calibration](#approach-b-implicit-calibration)
6. [Comparison & Recommendations](#comparison--recommendations)

---

## Problem Statement

### What is Model Calibration?

**Calibration** measures how well a model's confidence scores align with actual correctness rates:

- **Well-calibrated**: If model says 70% confident, it's correct 70% of the time
- **Poorly-calibrated**: Model is overconfident (90% confidence, 60% accuracy) or underconfident (50% confidence, 80% accuracy)

### Why Does Calibration Matter for LLM Routing?

Our LLM Router needs accurate confidence scores to make cost-effective routing decisions:

```
Query → SLM (cheap, fast) → Confidence?
  ├─ High confidence (>0.7) → Use SLM answer ✅
  └─ Low confidence (<0.7) → Escalate to LLM (expensive) 🔄
```

**Problem**: Current SLMs have **poor calibration** (RMS error ~0.45-0.50), leading to:
- **False escalations**: Correct answers with low confidence → unnecessary LLM calls
- **Missed errors**: Incorrect answers with high confidence → bad results

**Goal**: Improve calibration (reduce RMS error by ≥20%) without degrading answer quality.

---

## Baseline Measurement

### Methodology

To establish baseline calibration metrics, we:

1. **Selected 3 representative SLMs**:
   - `google/gemma-2-9b-it` (9B params) - larger model
   - `google/gemma-2-2b-it` (2B params) - target model for training
   - `Qwen/Qwen2.5-1.5B-Instruct` (1.5B params) - smallest model

2. **Ran inference on SimpleQA dataset**:
   - 1000 short-answer factual questions
   - Used vLLM for efficient batched inference on Modal (A100-40GB)
   - Extracted confidence from logprobs (geometric mean of token probabilities)

3. **Judged correctness with Gemini 2.0 Flash**:
   - Fast, accurate semantic equivalence checking
   - Async judging (100 concurrent requests) → ~3 minutes total
   - Binary labels: correct/incorrect

4. **Calculated calibration metrics**:
   - **RMS Calibration Error** (HLE's metric, β=100)
   - L1 Calibration Error
   - Accuracy vs Average Confidence gap

### Baseline Results

| Model | Accuracy | Avg Confidence | RMS Error | L1 Error | Assessment |
|-------|----------|----------------|-----------|----------|------------|
| **Gemma-9B** | 21.3% | 0.718 | **0.498** | 0.512 | Overconfident (71% conf, 21% acc) |
| **Gemma-2B** | 19.4% | 0.702 | **0.487** | 0.501 | Overconfident (70% conf, 19% acc) |
| **Qwen-1.5B** | 23.3% | 0.661 | **0.428** | 0.443 | Overconfident (66% conf, 23% acc) |

**Key Findings**:

1. **All models are overconfident**: Confidence 40-50 percentage points higher than accuracy
2. **Calibration worsens with size**: Larger models (9B) have worse calibration than smaller (1.5B)
3. **RMS error ~0.45-0.50**: Significant room for improvement
4. **Accuracy is low (19-23%)**: SimpleQA is challenging for SLMs

**Target**: Reduce Gemma-2B's RMS error from **0.487 → <0.39** (20% improvement)

---

## Data Generation Philosophy

### Core Principles

Our data generation follows these principles:

1. **Ground truth labels are essential**: We need actual correctness (not just confidence) to measure calibration
2. **Use strong judges, not gold labels**: Gemini 2.0 Flash provides accurate semantic equivalence checking at scale
3. **Focus on miscalibrated examples**: Training should emphasize where current calibration is worst
4. **Preserve answer quality**: Calibration improvements shouldn't degrade answer correctness

### Data Pipeline Overview

```
SimpleQA (1000 questions)
    ↓
SLM Inference (vLLM on Modal)
    → Answer + Confidence (from logprobs)
    ↓
Gemini Judging (async)
    → Correctness labels (binary)
    ↓
Training Data Generation
    ├─ DPO: Preference pairs (chosen vs rejected)
    └─ Implicit: (question, answer, is_correct) tuples
```

### Why This Approach?

**Advantages**:
- **Scalable**: Async judging processes 1000 examples in ~3 minutes
- **Accurate**: Gemini 2.0 Flash has high agreement with human judgments
- **Cost-effective**: $0.001/1000 examples for judging
- **Reproducible**: Deterministic pipeline with saved checkpoints

**Limitations**:
- **Judge quality**: Relies on Gemini's judgment accuracy (~95%+)
- **Distribution shift**: Training on SimpleQA may not generalize to all domains
- **Sample size**: 1000 examples may be insufficient for complex calibration patterns

---

## Approach A: Explicit Confidence DPO

### Rationale

**Direct Preference Optimization (DPO)** is a proven method for aligning models to preferences without RL instability. Our insight: **Well-calibrated confidence is a preference we can express as chosen vs rejected pairs**.

### Key Idea

Train the model to **explicitly generate confidence scores** that match correctness:

```
Prompt:
"Answer the following question and provide confidence (0.0-1.0):
Question: What is the capital of France?
Format: Answer: [your answer]\nConfidence: [0.0-1.0]"

Chosen (well-calibrated):
"Answer: Paris\nConfidence: 0.90"  ← Correct answer, high confidence ✅

Rejected (poorly-calibrated):
"Answer: Paris\nConfidence: 0.50"  ← Correct answer, low confidence ❌
```

### Data Generation Process

#### Step 1: Filter Miscalibrated Examples

From 1000 judged results, keep only examples with **|confidence - is_correct| > 0.3**:

```python
# Example filtering logic
for example in judged_results:
    original_conf = example['confidence']  # From logprobs
    is_correct = example['is_correct']      # From Gemini judge

    # Calculate calibration error
    individual_error = abs(original_conf - float(is_correct))

    # Only include if poorly calibrated
    if individual_error >= 0.3:
        include_in_training = True
```

**Rationale**: Focus training on examples where current calibration is worst.

**Result**: 897 / 1000 examples included (89.7%)

#### Step 2: Generate Preference Pairs

For each miscalibrated example, create a DPO pair:

```python
# Target confidence strategy: "binary"
target_confidence = 0.9 if is_correct else 0.1

# Chosen response (well-calibrated)
chosen = f"Answer: {answer}\nConfidence: {target_confidence:.2f}"

# Rejected response (poorly-calibrated)
rejected = f"Answer: {answer}\nConfidence: {original_confidence:.2f}"

dpo_pair = {
    "prompt": prompt,
    "chosen": chosen,
    "rejected": rejected,
}
```

**Why "binary" strategy (0.9 / 0.1)?**
- **Simplicity**: Clear separation between correct/incorrect
- **Strong signal**: Maximum gradient for DPO loss
- **Interpretability**: Easy to understand model's learned behavior

**Alternative "moderate" strategy (0.8 / 0.2)**:
- More conservative, less extreme confidence
- May generalize better to edge cases
- We default to "binary" for initial experiments

#### Step 3: DPO Training

Use standard TRL library with DPOConfig:

```python
# Training configuration
DPOConfig(
    beta=0.1,              # DPO temperature (controls preference strength)
    num_epochs=3,          # Standard fine-tuning duration
    learning_rate=1e-5,    # Conservative LR for LoRA
    per_device_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
)
```

**LoRA configuration** (for memory efficiency):
```python
LoraConfig(
    r=16,                  # Rank (standard for 2B models)
    lora_alpha=32,         # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

### Expected Outcomes

**Optimistic**:
- RMS error: 0.487 → **0.30** (38% reduction)
- Model learns to output well-calibrated explicit confidence
- No degradation in answer quality (confidence is separate from answer)

**Conservative**:
- RMS error: 0.487 → **0.39** (20% reduction)
- Some examples still poorly calibrated
- Model may default to extreme confidences (0.9/0.1)

**Risks**:
- **Format dependency**: Requires "Answer: X\nConfidence: Y" format at inference
- **Synthetic preferences**: Pairs are constructed, not from real comparisons
- **Overfit to binary strategy**: May not handle nuanced confidence well

### Advantages

1. ✅ **Uses proven DPO method** (standard library, well-studied)
2. ✅ **Interpretable outputs** (human-readable confidence scores)
3. ✅ **Clean separation** (answer vs confidence are independent)
4. ✅ **Easy to debug** (can inspect generated confidence directly)

### Disadvantages

1. ❌ **Requires format change** (must request confidence in prompt)
2. ❌ **Longer generation** (extra tokens for confidence)
3. ❌ **Backward incompatible** (can't use with existing logprobs-based routing)

---

## Approach B: Implicit Calibration

### Rationale

Instead of changing model output, **directly optimize internal representations** to be better calibrated. This preserves backward compatibility and requires no inference-time format changes.

### Key Idea

Add a **calibration penalty** to standard language modeling loss:

```python
# Standard LM loss
lm_loss = CrossEntropy(logits, labels)

# Calibration penalty
avg_prob = geometric_mean(token_probs)  # Model's internal confidence
calibration_loss = MSE(avg_prob, is_correct)  # Penalize deviation

# Combined loss
total_loss = lm_loss + λ * calibration_loss
```

**Intuition**:
- If answer is correct → maximize avg token probability (high confidence)
- If answer is incorrect → minimize avg token probability (low confidence)

### Data Generation Process

#### Step 1: Prepare Training Examples

Use ALL judged results (no filtering needed):

```python
# Training example format
{
    "input": "Question: {question}\nAnswer: {answer}",
    "is_correct": 1 or 0,  # Binary label from Gemini judge
}
```

**Why no filtering?**
- Calibration loss benefits from **full distribution** of examples
- Well-calibrated examples provide positive reinforcement
- All examples contribute to learning the confidence-correctness mapping

**Result**: 1000 training examples (100% of judged results)

#### Step 2: Custom Trainer Implementation

Implement custom `compute_loss` method:

```python
class CalibrationTrainer(Trainer):
    def __init__(self, calibration_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.calibration_weight = calibration_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract correctness labels
        is_correct = inputs.pop("is_correct").float()

        # Standard forward pass
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # Calculate calibration loss
        logits = outputs.logits
        probs = softmax(logits, dim=-1)
        token_probs = gather(probs, labels)

        # Geometric mean of token probabilities
        log_probs = log(token_probs + 1e-10)
        avg_log_prob = (log_probs * attention_mask).sum() / attention_mask.sum()
        avg_prob = exp(avg_log_prob)

        # MSE penalty
        calibration_loss = MSE(avg_prob, is_correct)

        # Combine losses
        total_loss = lm_loss + self.calibration_weight * calibration_loss

        return total_loss
```

**Key hyperparameter**: `calibration_weight = 0.1` (10% of total loss)
- Too high → degrades answer quality (model prioritizes confidence over correctness)
- Too low → insufficient calibration improvement
- 0.1 is a balanced starting point

#### Step 3: Training Configuration

```python
TrainingArguments(
    num_epochs=3,
    learning_rate=1e-5,
    per_device_batch_size=4,
    gradient_accumulation_steps=4,
)

# Same LoRA config as DPO approach
LoraConfig(r=16, lora_alpha=32, ...)
```

### Expected Outcomes

**Optimistic**:
- RMS error: 0.487 → **0.39** (20% reduction)
- Backward compatible (no format change)
- No degradation in answer quality

**Conservative**:
- RMS error: 0.487 → **0.44** (10% reduction)
- Calibration improvements are subtle
- May require tuning `calibration_weight`

**Risks**:
- **Indirect supervision**: Only penalizes average confidence, not per-decision
- **May affect answer quality**: Training changes token selection
- **Harder to debug**: Calibration improvements are implicit in logprobs

### Advantages

1. ✅ **No format change** (backward compatible)
2. ✅ **Efficient** (single training pass, no preference pair generation)
3. ✅ **Direct optimization** (targets internal representations)
4. ✅ **Uses all data** (no need to filter miscalibrated examples)

### Disadvantages

1. ❌ **Custom training code** (requires implementing custom Trainer)
2. ❌ **Hyperparameter sensitive** (calibration_weight needs tuning)
3. ❌ **Indirect signal** (only penalizes average confidence)
4. ❌ **Harder to debug** (can't inspect confidence directly)

---

## Comparison & Recommendations

### Side-by-Side Comparison

| Aspect | Explicit DPO (A) | Implicit Calibration (B) |
|--------|------------------|--------------------------|
| **Output format** | "Answer: X\nConfidence: Y" | Unchanged (logprobs) |
| **Training complexity** | Standard (TRL library) | Custom (custom Trainer) |
| **Data requirements** | Preference pairs (897) | Judged results (1000) |
| **Inference** | Parse text | Extract logprobs |
| **Backward compatible** | ❌ No | ✅ Yes |
| **Interpretability** | ✅ High (visible confidence) | ❌ Low (implicit) |
| **Expected improvement** | 20-40% (optimistic) | 10-20% (conservative) |
| **Development time** | Low (standard tools) | High (custom code) |
| **Risk** | Format dependency | May affect answer quality |

### Recommendation

**Primary approach**: **Explicit DPO (A)**

**Rationale**:
1. **Proven method**: DPO is well-studied with standard libraries
2. **Clear signal**: Explicit confidence is easier to optimize
3. **Interpretable**: Can inspect and analyze generated confidence
4. **Higher expected improvement**: 20-40% vs 10-20%

**Fallback**: **Implicit Calibration (B)**

**When to use**:
- If explicit DPO doesn't work well
- If backward compatibility is critical
- If format change is unacceptable for deployment

### Experimental Strategy

**Phase 1: Parallel training** (current)
- Train both approaches simultaneously
- Compare results on held-out test set
- Measure both calibration (RMS error) and accuracy

**Phase 2: Analysis** (after training)
- **Calibration curves**: Plot confidence vs accuracy bins
- **Per-category breakdown**: Analyze by SimpleQA topics
- **Failure analysis**: Identify where each approach fails

**Phase 3: Iteration** (if needed)
- Tune hyperparameters based on best-performing approach
- Generate more training data if needed
- Try hybrid approach (combine explicit + implicit)

---

## Cost Analysis

### Baseline Measurement

| Step | Cost | Time |
|------|------|------|
| vLLM inference (1000 examples × 3 models) | $6 | 30 min |
| Gemini judging (3000 examples, async) | $0.003 | 9 min |
| **Total** | **$6** | **~40 min** |

### Training (per approach)

| Resource | Cost | Time |
|----------|------|------|
| A100-40GB (Modal) | $5-10 | 1-2 hours |
| Data generation | $0 | 5 min (local) |
| **Total per approach** | **$5-10** | **~1-2 hours** |

### Both Approaches

**Total cost**: ~$10-20
**Total time**: ~2-4 hours (parallel training)

**Cost-effectiveness**: Extremely low compared to collecting human labels (~$100-500 for 1000 examples)

---

## Expected Timeline

```
Day 1 (Complete):
✅ Baseline inference (3 models × 1000 examples)
✅ Async judging (3000 examples)
✅ DPO data generation (897 pairs)
✅ Baseline metrics calculated

Day 2 (In Progress):
🔄 DPO training (1-2 hours)
🔄 Implicit calibration training (1-2 hours)

Day 3 (Next):
⏳ Post-training evaluation
⏳ Calibration analysis & comparison
⏳ Final report & recommendations
```

---

## Success Criteria

### Minimum Viable Improvement

- **Calibration**: RMS error reduction ≥10% (0.487 → 0.438)
- **Accuracy**: No degradation >5% (19.4% → >18.4%)
- **Confidence distribution**: Not collapsed (should span 0.2-0.9, not just 0.1/0.9)

### Target Improvement

- **Calibration**: RMS error reduction ≥20% (0.487 → 0.390)
- **Accuracy**: Maintained or improved (≥19.4%)
- **Generalization**: Works on held-out SimpleQA test set

### Stretch Goal

- **Calibration**: RMS error reduction ≥30% (0.487 → 0.341)
- **Accuracy**: Improved (≥21%)
- **Robustness**: Works across different question types and difficulties

---

## Appendix: Technical Details

### Confidence Extraction from Logprobs

```python
# Geometric mean of token probabilities
def extract_confidence(logprobs: List[float]) -> float:
    """Extract confidence from token logprobs (vLLM output)."""
    # Convert logprobs to probabilities
    probs = [math.exp(lp) for lp in logprobs]

    # Geometric mean
    log_mean = sum(math.log(p + 1e-10) for p in probs) / len(probs)
    confidence = math.exp(log_mean)

    return confidence
```

**Why geometric mean?**
- More sensitive to low-probability tokens (catches uncertainty)
- Aligns with model's generation process (multiplicative probabilities)
- Standard practice in literature

### RMS Calibration Error (HLE Metric)

```python
def calib_err(confidence: np.ndarray, correct: np.ndarray,
              p: str = '2', beta: int = 100) -> float:
    """
    Calculate calibration error using HLE's metric.

    Args:
        confidence: Model confidence scores [0, 1]
        correct: Binary correctness labels {0, 1}
        p: Norm type ('1' for L1, '2' for RMS)
        beta: Bin size for grouping examples

    Returns:
        Calibration error (lower is better)
    """
    # Implementation from evaluation/calibration.py
    # Groups examples by confidence bins, calculates |confidence - accuracy| per bin
    ...
```

**Why β=100?**
- Standard parameter from HLE benchmark
- Balances bin resolution vs statistical significance
- Works well for 1000-example datasets

---

## References

- **HLE Benchmark**: https://lastexam.ai (SimpleQA dataset)
- **DPO Paper**: "Direct Preference Optimization" (Rafailov et al., 2023)
- **Calibration Literature**: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
- **Modal Docs**: https://modal.com/docs (serverless GPU platform)

---

## Document Metadata

- **Created**: 2025-01-09
- **Last Updated**: 2025-01-09
- **Authors**: LLM Router Team
- **Status**: Active (training in progress)

---

*This document will be updated with post-training results and analysis once experiments complete.*
