# Model Calibration Strategies for LLM Router

## Overview

This document outlines various strategies for teaching language models to be well-calibrated - i.e., their confidence scores should align with actual correctness rates.

**Problem**: Baseline gemma-2-2b-it model has 19.4% accuracy but 75-90% confidence → severe overconfidence (RMS error: 0.3888)

**Goal**: Reduce calibration error while maintaining or improving accuracy.

---

## Strategies We're Currently Testing

### 1. **DPO (Direct Preference Optimization) - Explicit Confidence**

**Approach**: Train model to output confidence explicitly in text format.

**Output Format**:
```
Answer: Paris
Confidence: 0.85
```

**Training Method**: Preference learning with (chosen, rejected) pairs.

#### Implementation Details

**v1 (FAILED - Same Answer Problem)**:
```python
# For ALL answers (incorrect approach)
chosen = "Answer: Paris\nConfidence: 0.90"     # Target confidence
rejected = "Answer: Paris\nConfidence: 0.62"   # Original confidence
```

**Problem**: Model learned to output flat 0.5 confidence as compromise.

**v2 (Current - Ground Truth + Randomized Nudging)**:
```python
# For CORRECT answers (19.4%)
actual_nudge = 0.2 + random.uniform(-0.1, 0.1)  # Randomized: 0.1-0.3
chosen = f"Answer: {model_answer}\nConfidence: {original + actual_nudge:.2f}"
rejected = f"Answer: {model_answer}\nConfidence: {original:.2f}"

# For INCORRECT answers (80.6%)
actual_nudge = 0.2 + random.uniform(-0.1, 0.1)  # Randomized: 0.1-0.3
chosen = f"Answer: {ground_truth}\nConfidence: 0.90"  # Ground truth!
rejected = f"Answer: {model_answer}\nConfidence: {original - actual_nudge:.2f}"
```

**Key Improvements**:
1. Uses ground truth answers for incorrect predictions → teaches accuracy
2. Randomized nudging (±0.1 variance) → smoother learning signal
3. Different answers in chosen/rejected → model can learn

**Data**: 1000 pairs (194 correct, 806 incorrect)

**Expected Outcome**: Improved calibration AND accuracy

**Status**: ✅ Training now (`google_gemma-2-2b-it_dpo_pairs_v2.jsonl`)

**Files**:
- Training: `scripts/modal_dpo_training.py`
- Data generation: `data/generate_improved_dpo_pairs.py`
- Data: `data/slm_baseline/google_gemma-2-2b-it_dpo_pairs_v2.jsonl`

---

### 2. **Implicit Calibration - Custom Loss Function**

**Approach**: Add calibration penalty to training loss without changing output format.

**Output Format**: Normal model output (no confidence score visible)

**Confidence Extraction**: Geometric mean of token probabilities (logprobs)

#### Implementation Details

**Custom Loss**:
```python
total_loss = lm_loss + calibration_weight * calibration_loss

# Where:
calibration_loss = MSE(avg_token_prob, is_correct)

# avg_token_prob = geometric mean of P(token|context)
log_probs = torch.log(token_probs + 1e-10)
avg_log_prob = (log_probs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
avg_prob = torch.exp(avg_log_prob)
```

**Calibration Penalty**:
- If correct (is_correct=1): penalizes low avg_prob → pushes toward high confidence
- If incorrect (is_correct=0): penalizes high avg_prob → pushes toward low confidence

**Hyperparameters**:
- `calibration_weight`: 0.1 (10% of total loss)
- `learning_rate`: 1e-5
- `epochs`: 3

**Data**: 1000 judged examples (194 correct, 806 incorrect)

**Expected Outcome**: Improved calibration, accuracy stays ~19.4% (calibration-only, no ground truth)

**Status**: ✅ Training now

**Files**:
- Training: `scripts/modal_implicit_calibration_training.py`
- Uses CalibrationTrainer with custom compute_loss()

---

## Alternative Strategies (Not Yet Implemented)

### 3. **Standard SFT with Confidence**

**Approach**: Simple supervised fine-tuning on (question, answer+confidence) pairs.

**Difference from DPO**: No preference learning, just direct supervision.

**Training Data Format**:
```json
{
  "input": "Question: What is the capital of France?",
  "output": "Answer: Paris\nConfidence: 0.95"
}
```

**Data Generation Strategy**:
```python
# For correct answers
output = f"Answer: {model_answer}\nConfidence: 0.90"

# For incorrect answers
output = f"Answer: {ground_truth}\nConfidence: 0.90"
# OR teach low confidence for wrong answers:
output = f"Answer: {model_answer}\nConfidence: 0.10"
```

**Pros**:
- Simpler than DPO (no preference pairs)
- Can teach both accuracy and calibration

**Cons**:
- May be less effective than preference learning
- Harder to teach nuanced confidence levels

**Implementation Complexity**: ⭐⭐ (Easy - standard Trainer)

**Expected Time**: 1-2 hours training

---

### 4. **Uncertainty Token Approach**

**Approach**: Train model to output discrete uncertainty markers instead of numeric scores.

**Output Format**:
```
Answer: Paris [HIGH_CONFIDENCE]
Answer: Berlin [MEDIUM_CONFIDENCE]
Answer: London [LOW_CONFIDENCE]
```

**Conversion to Scores**:
```python
token_to_score = {
    "[HIGH_CONFIDENCE]": 0.9,
    "[MEDIUM_CONFIDENCE]": 0.5,
    "[LOW_CONFIDENCE]": 0.1,
}
```

**Training Data**:
```python
# Bin confidence scores into discrete levels
if is_correct:
    if original_conf > 0.7:
        token = "[HIGH_CONFIDENCE]"
    elif original_conf > 0.4:
        token = "[MEDIUM_CONFIDENCE]"
    else:
        token = "[LOW_CONFIDENCE]"
```

**Pros**:
- Easier for model to learn discrete tokens vs continuous scores
- More interpretable
- Can use more granular bins (e.g., 5 or 10 levels)

**Cons**:
- Loses precision (discrete vs continuous)
- Still requires threshold tuning

**Implementation Complexity**: ⭐⭐ (Easy - standard SFT)

**Expected Time**: 1-2 hours training

---

### 5. **Temperature Calibration (Post-hoc)**

**Approach**: Learn a temperature parameter to scale logits without retraining model.

**Method**:
```python
# Normal inference
logits = model(input)

# Calibrated inference
calibrated_logits = logits / temperature

# Find optimal temperature on validation set
def find_optimal_temperature(model, val_data):
    best_temp = 1.0
    best_ece = float('inf')

    for temp in np.arange(0.5, 3.0, 0.1):
        ece = calculate_ece(model, val_data, temperature=temp)
        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    return best_temp
```

**Calibration Metrics**:
- Expected Calibration Error (ECE)
- RMS Calibration Error (our current metric)

**Pros**:
- No retraining needed
- Fast to compute (just grid search)
- Preserves model accuracy

**Cons**:
- Single global parameter (not example-specific)
- May not fix severe miscalibration
- Requires held-out calibration set

**Implementation Complexity**: ⭐ (Very Easy - just inference)

**Expected Time**: 5-10 minutes

---

### 6. **Platt Scaling / Isotonic Regression**

**Approach**: Train a separate calibration model to map uncalibrated scores → calibrated probabilities.

**Method**:

**Platt Scaling** (Logistic Regression):
```python
from sklearn.linear_model import LogisticRegression

# Collect uncalibrated scores on validation set
uncalibrated_scores = []
labels = []

for example in val_data:
    score = model.get_confidence(example)  # From logprobs
    label = is_correct(example)
    uncalibrated_scores.append(score)
    labels.append(label)

# Train calibration model
calibrator = LogisticRegression()
calibrator.fit(np.array(uncalibrated_scores).reshape(-1, 1), labels)

# Calibrated inference
calibrated_score = calibrator.predict_proba([[uncalibrated_score]])[0][1]
```

**Isotonic Regression** (Non-parametric):
```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(uncalibrated_scores, labels)

# Calibrated inference
calibrated_score = calibrator.predict([uncalibrated_score])[0]
```

**Pros**:
- No retraining needed
- Can fix non-linear miscalibration
- Isotonic regression is more flexible than Platt scaling

**Cons**:
- Requires held-out calibration set
- Isotonic can overfit with small data
- Doesn't improve base model accuracy

**Implementation Complexity**: ⭐⭐ (Easy - sklearn)

**Expected Time**: 5-10 minutes

---

### 7. **Ensemble Disagreement as Uncertainty**

**Approach**: Use disagreement among multiple models/runs as confidence signal.

**Method**:
```python
# Sample multiple outputs with different seeds/temperatures
answers = []
for i in range(5):
    answer = model.generate(prompt, temperature=0.7, seed=i)
    answers.append(answer)

# Confidence = agreement rate
unique_answers = set(answers)
most_common = max(unique_answers, key=answers.count)
confidence = answers.count(most_common) / len(answers)

# Example:
# ["Paris", "Paris", "Paris", "London", "Paris"] → confidence = 0.8
```

**Pros**:
- No training needed
- Works with any model
- Empirically robust

**Cons**:
- 5x inference cost
- May still be miscalibrated (just shifted)
- Doesn't improve base accuracy

**Implementation Complexity**: ⭐ (Very Easy - just inference)

**Expected Time**: Immediate (just run inference)

---

### 8. **Self-Consistency with Chain-of-Thought**

**Approach**: Generate reasoning paths, use consistency across paths as confidence.

**Method**:
```python
# Prompt model to show reasoning
prompt = """Question: What is the capital of France?

Let's think step by step:
1. France is a country in Europe
2. The capital is the main city
3. """

# Sample multiple reasoning paths
paths = []
for i in range(5):
    path = model.generate(prompt, temperature=0.7, seed=i)
    final_answer = extract_final_answer(path)
    paths.append(final_answer)

# Confidence = consistency
confidence = max_count(paths) / len(paths)
```

**Pros**:
- Leverages model's reasoning ability
- Can improve accuracy via majority voting
- No training needed

**Cons**:
- Requires CoT capability (may not work for small models)
- Higher inference cost
- Longer generation time

**Implementation Complexity**: ⭐⭐ (Moderate - prompt engineering)

**Expected Time**: Immediate (just run inference)

---

## Comparison Matrix

| Strategy | Training Needed | Changes Output Format | Improves Accuracy | Complexity | Expected RMS Improvement |
|----------|----------------|----------------------|-------------------|------------|-------------------------|
| **DPO v2** ✅ | Yes (1-2h) | Yes (adds confidence) | Yes (ground truth) | ⭐⭐⭐ | High (0.39 → ~0.10) |
| **Implicit Calibration** ✅ | Yes (1-2h) | No | No | ⭐⭐⭐⭐ | Medium (0.39 → ~0.20) |
| **Standard SFT** | Yes (1-2h) | Yes (adds confidence) | Yes (ground truth) | ⭐⭐ | High (0.39 → ~0.15) |
| **Uncertainty Tokens** | Yes (1-2h) | Yes (adds tokens) | Yes (ground truth) | ⭐⭐ | Medium (0.39 → ~0.20) |
| **Temperature Calibration** | No | No | No | ⭐ | Low (0.39 → ~0.30) |
| **Platt/Isotonic** | No (calibrator) | No | No | ⭐⭐ | Low-Medium (0.39 → ~0.25) |
| **Ensemble Disagreement** | No | No | No | ⭐ | Low (0.39 → ~0.30) |
| **Self-Consistency CoT** | No | Yes (adds reasoning) | Yes (voting) | ⭐⭐ | Medium (0.39 → ~0.20) |

---

## Recommendations

### Immediate Next Steps (After Current Training)
1. ✅ **Evaluate DPO v2 and Implicit** - Already training
2. Compare results to baseline (0.3888 RMS)
3. Pick winner for production

### Quick Experiments (If Current Approaches Fail)
1. **Temperature Calibration** - 5 minutes, no training
2. **Ensemble Disagreement** - 10 minutes, works with current model
3. **Platt Scaling** - 10 minutes, simple calibration

### If We Have More Time
1. **Standard SFT with Confidence** - Simpler than DPO, may work better
2. **Uncertainty Tokens** - Interesting middle ground
3. **Self-Consistency CoT** - Leverages reasoning ability

### Long-term Research
1. Combine approaches (e.g., Implicit + Temperature)
2. Active learning: generate training data for most miscalibrated examples
3. Multi-task learning: train on calibration + other tasks

---

## Evaluation Metrics

### Primary Metric
- **RMS Calibration Error** (HLE with β=100)
  - Baseline: 0.3888
  - Target: <0.15

### Secondary Metrics
- **Accuracy**: 19.4% → maintain or improve
- **ECE (Expected Calibration Error)**: Standard calibration metric
- **Sharpness**: Average confidence (want confident when correct)
- **Per-bin analysis**: Identify which confidence ranges are most miscalibrated

---

## Implementation Files

### Current
- `scripts/modal_dpo_training.py` - DPO training
- `scripts/modal_implicit_calibration_training.py` - Implicit calibration
- `scripts/modal_evaluate_calibration.py` - Evaluation script
- `data/generate_improved_dpo_pairs.py` - DPO data generation

### To Create (for alternatives)
- `scripts/modal_sft_calibration.py` - Standard SFT
- `scripts/temperature_calibration.py` - Temperature tuning
- `scripts/platt_calibration.py` - Platt scaling
- `evaluation/ensemble_confidence.py` - Ensemble disagreement

---

## References

### Papers
1. **DPO**: "Direct Preference Optimization" (Rafailov et al., 2023)
2. **Temperature Calibration**: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
3. **Platt Scaling**: "Probabilistic Outputs for SVMs" (Platt, 1999)
4. **Self-Consistency**: "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2022)

### Related Work
- Calibration in deep learning: https://arxiv.org/abs/1706.04599
- Uncertainty estimation: https://arxiv.org/abs/1902.07801
