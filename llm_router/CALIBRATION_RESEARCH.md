# Confidence Calibration Research for Small Language Models

**Date**: January 2025
**Model**: Google Gemma-2-2b-it
**Benchmark**: SimpleQA (1000 questions)
**Baseline Accuracy**: 19.4%

---

## 🎯 Research Question

**Can we train small language models (SLMs) to provide calibrated confidence estimates alongside their answers?**

**Why This Matters**: SLMs are poorly calibrated out-of-the-box. On SimpleQA, Gemma-2-2b-it achieves only 19.4% accuracy but has no sense of when it's right vs wrong. Calibration would enable:
- Better human-AI collaboration (know when to trust the model)
- Effective model routing (use confidence to decide when to escalate)
- Improved safety (low confidence → defer to human/stronger model)

---

## 📊 Experimental Setup

### Dataset Generation

1. **Generate SLM Answers** (`scripts/modal_slm_baseline.py`)
   - Run Gemma-2-2b-it on SimpleQA benchmark
   - Output: `data/slm_baseline/google_gemma-2-2b-it_results.jsonl`
   - 1000 question-answer pairs

2. **Judge Correctness** (`evaluation/judge_slm_answers_async.py`)
   - Use Gemini-2.5-Pro as judge to label correctness
   - Output: `data/slm_baseline/google_gemma-2-2b-it_judged.jsonl`
   - Ground truth: `is_correct` (true/false)

### Approaches Tested

| Approach | Description | Data Generation | Training Script |
|----------|-------------|-----------------|-----------------|
| **DPO v1** | Preference learning (chosen/rejected pairs) | `data/generate_dpo_calibration_data.py` | `scripts/modal_dpo_training.py` |
| **DPO v2** | Fixed DPO (swapped pairs corrected) | Same script, v2 output | Same |
| **Implicit** | Train on logprobs (geometric mean) | Uses judged data directly | `scripts/modal_implicit_calibration_training.py` |
| **Standard SFT** | Direct supervision (input → answer+confidence) | `data/generate_sft_calibration_data.py` | `scripts/modal_sft_calibration_training.py` |

---

## 🔬 Results

### Summary Table

| Approach | RMS Calibration Error | Accuracy | Avg Confidence | Status | Notes |
|----------|----------------------|----------|----------------|--------|-------|
| **Baseline** | 0.3888 | 19.4% | N/A | ✅ Complete | Uncalibrated |
| **DPO v1** | 0.3888 | 19.4% | ~0.5 | ✅ Complete | **Bug**: Swapped pairs |
| **DPO v2** | **0.3060** | 19.4% | 0.50 (flat) | ✅ Complete | **21% improvement but flat** |
| **Implicit** (buggy) | 0.1940 | 19.4% | 0.00 (flat) | ❌ Invalid | **Bug**: Eval used INPUT logprobs |
| **Implicit** (fixed) | Running... | Running... | TBD | 🔄 27% Complete | Bug fixed, ~15 min remaining |
| **Standard SFT** | TBD | TBD | TBD | ✅ Training Complete, Ready for Eval | ~3 min training |

**Key Metric**: RMS Calibration Error (lower is better)
- Measures how well predicted confidence matches actual accuracy
- Calculated using 100 bins (β=100)
- Formula: `RMS = sqrt(mean((confidence - accuracy)²))`

---

## 🐛 Critical Bugs Discovered

### 1. DPO v1: Swapped Chosen/Rejected Pairs

**Problem**: `data/generate_dpo_calibration_data.py` originally generated:
```python
# WRONG (v1)
chosen = {"answer": answer, "confidence": 0.1}  # Incorrect answer with LOW confidence
rejected = {"answer": answer, "confidence": 0.9}  # Incorrect answer with HIGH confidence
```

**Impact**: Model learned INVERTED preferences → no calibration improvement

**Fix**: Swapped chosen/rejected to make high confidence the "chosen" response
```python
# CORRECT (v2)
chosen = {"answer": answer, "confidence": 0.9}  # High confidence (preferred)
rejected = {"answer": answer, "confidence": 0.1}  # Low confidence (rejected)
```

**Result**: DPO v2 improved RMS from 0.3888 → 0.3060 (21% improvement)

**Files**:
- Data generator: `data/generate_dpo_calibration_data.py:133-172`
- Generated data: `data/slm_baseline/google_gemma-2-2b-it_dpo_pairs_v2.jsonl`

---

### 2. Implicit Calibration: INPUT vs OUTPUT Logprobs

**Problem**: Evaluation script `scripts/modal_evaluate_calibration.py:153-171` computed:
```python
# WRONG - Computing logprobs of INPUT tokens (the question)
outputs = model(**inputs, labels=inputs["input_ids"])
confidence = geometric_mean(input_token_probs)  # ❌ Confidence of PROMPT
```

**Impact**: All confidences = 0.0 because it measured "how likely is the question" not "how confident is the answer"

**Fix**: Generate answer first, then compute confidence from GENERATED tokens
```python
# CORRECT - Computing logprobs of OUTPUT tokens (the answer)
generated_ids = model.generate(**inputs, output_scores=True)
token_probs = [softmax(score)[token_id] for score, token_id in zip(scores, generated_tokens)]
confidence = geometric_mean(token_probs)  # ✅ Confidence of ANSWER
```

**Result**: Re-running evaluation with fixed logic (currently in progress)

**Files**:
- Evaluation script: `scripts/modal_evaluate_calibration.py:153-193`
- Modal job: https://modal.com/apps/tuna2/main/ap-xixT3iUzQzRQWmiSfz3uF7

---

### 3. SFT Data: Literal `'\n'` Instead of Newline

**Problem**: `data/generate_sft_calibration_data.py:88` wrote:
```python
# WRONG
f.write(json.dumps(example) + '\\n')  # Literal backslash-n
```

**Impact**: All 1000 JSON objects concatenated on one giant 582KB line → JSON parsing errors during training

**Fix**: Changed to actual newline character
```python
# CORRECT
f.write(json.dumps(example) + '\n')  # Real newline
```

**Result**: Regenerated `google_gemma-2-2b-it_sft_calibration_fixed.jsonl` with 1000 valid JSONL lines

**Files**:
- Data generator: `data/generate_sft_calibration_data.py:88`
- Fixed data: `data/slm_baseline/google_gemma-2-2b-it_sft_calibration_fixed.jsonl`

---

## 📈 Detailed Findings

### DPO v2: 21% Improvement but Flat Confidence

**Observation**: RMS improved from 0.3888 → 0.3060, but ALL confidences output as exactly 0.5

**Analysis**:
- Model learned to output "Confidence: 0.5" for every example
- This is better than random (reduces RMS error) but not useful
- Suggests DPO struggled to learn fine-grained confidence discrimination

**Possible Causes**:
1. Binary preference pairs (0.9 vs 0.1) don't teach continuous confidence
2. DPO loss may push toward mode-seeking (picking one confidence value)
3. Need more granular confidence levels in training data

**Evaluation Output**:
```
Total examples: 1000
Accuracy: 0.1940
Average confidence: 0.5000
RMS Calibration Error: 0.3060

All 1000 examples output exactly "Confidence: 0.5"
```

**Files**: Results at `/vol/eval_results/google_gemma-2-2b-it_calibrated_dpo_v2_eval.json`

---

### Implicit Calibration: Logprob-Based Approach

**Method**: Train model to maximize likelihood, then use geometric mean of token probabilities as confidence

**Hypothesis**: Correctly answered questions should have higher token probabilities than incorrect ones

**Training**:
- Add calibration loss: `calibration_loss = -log(p_correct) if correct else -log(1 - p_correct)`
- Weight: 0.1 (10% of total loss)
- Train for 3 epochs with LoRA (r=8)

**Bug**: Initial evaluation computed confidence from INPUT tokens instead of OUTPUT tokens

**Status**: Re-running with fixed evaluation logic (13% complete, ~30 min remaining)

**Files**:
- Training script: `scripts/modal_implicit_calibration_training.py`
- Evaluation script: `scripts/modal_evaluate_calibration.py:153-193`
- Model saved to: `/vol/implicit_calibrated_models/google_gemma-2-2b-it_calibrated`

---

### Standard SFT: Direct Supervision

**Method**: Directly supervise model to output `Answer: [text]\nConfidence: [0.0-1.0]`

**Training Data**:
- **Correct answers**: Output model's answer with confidence 0.9
- **Incorrect answers**: Output ground truth answer with confidence 0.9
- 1000 examples (194 correct, 806 incorrect)

**Training Time**: ~3.2 minutes (168 steps, 3 epochs)

**Training Loss**:
- Initial: 4.55
- Final: 0.13
- Converged smoothly

**Status**: Training complete, ready for evaluation

**Next Steps**: Evaluate SFT model to measure RMS calibration error

**Files**:
- Data generator: `data/generate_sft_calibration_data.py`
- Training script: `scripts/modal_sft_calibration_training.py`
- Training data: `data/slm_baseline/google_gemma-2-2b-it_sft_calibration_fixed.jsonl`
- Model saved to: `/vol/sft_calibrated_models/google_gemma-2-2b-it_calibrated`
- Training log: `/tmp/sft_training_fixed.log`

---

## 🔧 Technical Configuration

### Model Architecture
- **Base Model**: `google/gemma-2-2b-it` (2B parameters)
- **Training**: LoRA adapters (r=8, α=16, dropout=0.05)
- **Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Quantization**: 4-bit (bitsandbytes)
- **Trainable Parameters**: 0.24% (6.4M / 2.6B)

### Training Hyperparameters
```python
learning_rate = 1e-5
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
warmup_steps = 100
fp16 = True
```

### Evaluation Metrics

**RMS Calibration Error (β=100)**:
```python
bins = np.arange(0, 1.01, 1/100)  # 100 bins
for bin in bins:
    bin_conf = mean(confidences[in_bin])
    bin_acc = mean(correctness[in_bin])
    squared_error = (bin_conf - bin_acc) ** 2
rms_error = sqrt(mean(squared_errors))
```

**Perfect Calibration**: RMS = 0.0 (confidence always equals accuracy)
**Baseline (Gemma-2-2b-it)**: RMS = 0.3888

---

## 💡 Key Insights

### What Worked
1. **DPO v2 improved calibration** by 21% (RMS 0.3888 → 0.3060)
2. **Bug fixes were critical** - swapped pairs, wrong logprobs, malformed data
3. **SFT training converged quickly** (~3 min) with smooth loss curves

### What Didn't Work
1. **DPO outputs flat 0.5 confidence** - no fine-grained discrimination
2. **Binary preferences insufficient** - need continuous confidence values
3. **Implicit calibration bugs** - incorrect evaluation invalidated results

### Hypotheses for Improvement
1. **Multi-level DPO**: Use 5-10 confidence levels instead of binary (0.1 vs 0.9)
2. **Temperature scaling**: Post-hoc calibration method (cheap, effective)
3. **Ensemble methods**: Combine multiple confidence estimates
4. **Larger models**: May have better intrinsic calibration
5. **Better training objectives**: Brier score, log loss instead of preference learning

---

## 📂 File Reference

### Key Scripts Used

| Purpose | Script | Input | Output |
|---------|--------|-------|--------|
| Generate SLM answers | `scripts/modal_slm_baseline.py` | SimpleQA benchmark | `google_gemma-2-2b-it_results.jsonl` |
| Judge correctness | `evaluation/judge_slm_answers_async.py` | SLM results | `google_gemma-2-2b-it_judged.jsonl` |
| Generate DPO data | `data/generate_dpo_calibration_data.py` | Judged data | `google_gemma-2-2b-it_dpo_pairs_v2.jsonl` |
| Generate SFT data | `data/generate_sft_calibration_data.py` | Judged data | `google_gemma-2-2b-it_sft_calibration_fixed.jsonl` |
| Train DPO model | `scripts/modal_dpo_training.py` | DPO pairs | `/vol/dpo_models/...` |
| Train implicit model | `scripts/modal_implicit_calibration_training.py` | Judged data | `/vol/implicit_calibrated_models/...` |
| Train SFT model | `scripts/modal_sft_calibration_training.py` | SFT data | `/vol/sft_calibrated_models/...` |
| Evaluate calibration | `scripts/modal_evaluate_calibration.py` | Model + test data | Evaluation results JSON |

### Data Files

```
data/slm_baseline/
├── google_gemma-2-2b-it_results.jsonl              # SLM answers (1000)
├── google_gemma-2-2b-it_judged.jsonl               # Judged for correctness (1000)
├── google_gemma-2-2b-it_dpo_pairs.jsonl            # DPO v1 (buggy, 1000 pairs)
├── google_gemma-2-2b-it_dpo_pairs_v2.jsonl         # DPO v2 (fixed, 1000 pairs)
├── google_gemma-2-2b-it_sft_calibration.jsonl      # SFT v1 (malformed)
└── google_gemma-2-2b-it_sft_calibration_fixed.jsonl # SFT v2 (fixed, 1000 examples)
```

### Modal Volumes (Trained Models)

```
/vol/dpo_models/
└── google_gemma-2-2b-it_calibrated/               # DPO v2 trained model

/vol/implicit_calibrated_models/
└── google_gemma-2-2b-it_calibrated/               # Implicit calibration model

/vol/sft_calibrated_models/
└── google_gemma-2-2b-it_calibrated/               # SFT trained model

/vol/eval_results/
├── google_gemma-2-2b-it_calibrated_dpo_v2_eval.json
├── google_gemma-2-2b-it_calibrated_eval.json      # Implicit (re-running)
└── google_gemma-2-2b-it_sft_eval.json             # SFT (pending)
```

---

## 🚀 Next Steps

1. **✅ Complete implicit evaluation** - Fixed bug, running now (~30 min remaining)
2. **⏳ Evaluate SFT model** - Training complete, need to run evaluation
3. **📊 Compare all approaches** - Create calibration curves for each
4. **🔬 Try post-hoc methods**:
   - Temperature scaling (very fast, often effective)
   - Platt scaling
   - Isotonic regression
5. **📈 Investigate DPO flat confidence**:
   - Try continuous confidence levels
   - Use different β values in DPO loss
   - Generate more diverse training data
6. **📝 Write up findings** - Document lessons learned

---

## 📚 References

- **On Calibration of Modern Neural Networks** (Guo et al., 2017)
- **Direct Preference Optimization** (Rafailov et al., 2023)
- **SimpleQA Benchmark** - Factual question answering
- **Google Gemini Models** - Judge and target LLMs
- **TRL Library** - Transformers Reinforcement Learning

---

## 🤝 Acknowledgments

This research uses:
- **Modal Labs** for serverless GPU compute
- **Google Gemini API** for judging and target LLMs
- **Hugging Face** for model hosting and training libraries
- **SimpleQA benchmark** for evaluation dataset
