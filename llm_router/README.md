# LLM Router: Performance-Aligned Model Selection

A flexible, efficient router that intelligently directs queries to the optimal LLM (e.g., Gemini Flash vs Pro) based on query complexity, reducing costs while maintaining accuracy.

## 🎯 Project Overview

This project explores two related research questions:

1. **Primary**: Can a small language model (SLM) learn to route queries to appropriate LLMs based on complexity?
2. **Secondary**: Can we train SLMs to provide calibrated confidence estimates for their answers?

## 📊 Current Status: Model Calibration Research

We are currently investigating **confidence calibration** for small language models. See [CALIBRATION_RESEARCH.md](CALIBRATION_RESEARCH.md) for detailed findings.

**Quick Summary:**
- **Problem**: SLMs are poorly calibrated (RMS error: 0.3888 baseline)
- **Approaches Tested**: DPO, Implicit Calibration, Standard SFT
- **Best Result So Far**: DPO v2 (RMS 0.3060, 21% improvement) but outputs flat 0.5 confidence
- **Status**: Training SFT and re-evaluating implicit calibration with bug fixes

---

## 🏗️ Architecture

```
Query → Router (Gemma-2B) → Policy → Model Map → Target LLM → Answer
                ↓
    Standard_Query  → gemini-2.5-flash
    Complex_Query   → gemini-2.5-pro
    Ambiguous_Query → gemini-2.5-flash
```

## 📦 Installation

```bash
# Clone repository
cd llm_router

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-key"
export MODAL_TOKEN_ID="your-modal-token"
export MODAL_TOKEN_SECRET="your-modal-secret"
```

---

## 🔬 Research Workflows

### A. Model Calibration Experiments

**Goal**: Train SLMs to output calibrated confidence scores alongside their answers.

#### 1. Generate Baseline SLM Answers

```bash
# Run SLM inference on benchmark (SimpleQA)
modal run scripts/modal_slm_baseline.py \
  --model google/gemma-2-2b-it

# Output: data/slm_baseline/google_gemma-2-2b-it_results.jsonl
```

#### 2. Judge Answers for Correctness

```bash
# Use Gemini Pro to judge correctness
export GEMINI_API_KEY="your-key"
python3 evaluation/judge_slm_answers_async.py \
  --input data/slm_baseline/google_gemma-2-2b-it_results.jsonl \
  --output data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --max-concurrent 100

# Output: Judged dataset with is_correct labels
```

#### 3. Generate Calibration Training Data

**DPO (Explicit Confidence):**
```bash
python3 data/generate_dpo_calibration_data.py \
  --judged-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --output data/slm_baseline/google_gemma-2-2b-it_dpo_pairs_v2.jsonl

# Output: (chosen, rejected) preference pairs
```

**SFT (Direct Supervision):**
```bash
python3 data/generate_sft_calibration_data.py \
  --judged-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --output data/slm_baseline/google_gemma-2-2b-it_sft_calibration.jsonl

# Output: (input, output) supervised pairs
```

#### 4. Train Calibrated Models

**DPO Training:**
```bash
modal run scripts/modal_dpo_training.py \
  --model google/gemma-2-2b-it \
  --dpo-data data/slm_baseline/google_gemma-2-2b-it_dpo_pairs_v2.jsonl \
  --learning-rate 1e-5 \
  --num-epochs 3

# Saves to: /vol/dpo_models/google_gemma-2-2b-it_calibrated
```

**Implicit Calibration (Logprob-based):**
```bash
modal run scripts/modal_implicit_calibration_training.py \
  --model google/gemma-2-2b-it \
  --judged-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --learning-rate 1e-5 \
  --num-epochs 3 \
  --calibration-weight 0.1

# Saves to: /vol/implicit_calibrated_models/google_gemma-2-2b-it_calibrated
```

**Standard SFT:**
```bash
modal run scripts/modal_sft_calibration_training.py \
  --model google/gemma-2-2b-it \
  --sft-data data/slm_baseline/google_gemma-2-2b-it_sft_calibration.jsonl \
  --learning-rate 1e-5 \
  --num-epochs 3

# Saves to: /vol/sft_models/google_gemma-2-2b-it_calibrated
```

**RL-Based Calibration (REINFORCE):**
```bash
modal run scripts/modal_rl_calibration_vllm.py \
  --model google/gemma-2-2b-it \
  --judged-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --learning-rate 1e-5 \
  --num-epochs 3 \
  --lambda-sharp 0.3 \
  --lambda-extreme 0.2

# Anti-collapse rewards with REINFORCE policy gradient
# Saves to: /vol/rl_calibrated_models/google_gemma-2-2b-it_rl_vllm
```

**GRPO Calibration (Group Relative Policy Optimization):**
```bash
modal run scripts/modal_grpo_calibration_vllm.py \
  --model google/gemma-2-2b-it \
  --judged-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --learning-rate 1e-5 \
  --num-epochs 3 \
  --lambda-sharp 0.3 \
  --lambda-extreme 0.2

# Variance-reduced RL with batch-normalized rewards
# Saves to: /vol/grpo_calibrated_models/google_gemma-2-2b-it_grpo_vllm
```

#### 5. Evaluate Calibration

```bash
# Evaluate DPO model
modal run scripts/modal_evaluate_calibration.py \
  --model-path /vol/dpo_models/google_gemma-2-2b-it_calibrated \
  --test-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --model-type dpo

# Evaluate implicit calibration model
modal run scripts/modal_evaluate_calibration.py \
  --model-path /vol/implicit_calibrated_models/google_gemma-2-2b-it_calibrated \
  --test-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --model-type implicit

# Evaluate REINFORCE (RL) model
modal run scripts/modal_evaluate_rl_calibration.py \
  --test-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --model-type rl

# Evaluate GRPO model
modal run scripts/modal_evaluate_rl_calibration.py \
  --test-data data/slm_baseline/google_gemma-2-2b-it_judged.jsonl \
  --model-type grpo

# Outputs: RMS calibration error, ECE, accuracy, confidence distribution
```

---

### B. Router Training (Original Goal)

#### 1. Generate Oracle Dataset

```bash
python scripts/generate_oracle_dataset.py \
  --benchmark simpleqa \
  --input-file ./data/simpleqa.jsonl \
  --output-dir ./data/generated \
  --limit 1000 \
  --gemini-api-key $GEMINI_API_KEY

# Output: Oracle dataset with routing policies
```

#### 2. Prepare Training Data

```bash
python scripts/prepare_training_data.py \
  --oracle-file ./data/generated/oracle_dataset.jsonl \
  --output-dir ./data/training

# Output: SFT and DPO training files
```

#### 3. Train Router

```bash
modal run training/modal_train_router.py \
  --model-name google/gemma-2b \
  --sft-data-path ../data/training/sft_train.jsonl \
  --dpo-data-path ../data/training/dpo_train.jsonl \
  --sft-epochs 3 \
  --dpo-epochs 1

# Output: Trained router checkpoint
```

#### 4. Test Router

```bash
# Routing only
python scripts/test_router.py \
  --model-path ./router_checkpoints/router_final \
  --query "What is the wingspan of a Pteranodon?"

# End-to-end (routing + answer)
python scripts/test_router.py \
  --model-path ./router_checkpoints/router_final \
  --query "What is the wingspan of a Pteranodon?" \
  --end-to-end \
  --gemini-api-key $GEMINI_API_KEY
```

---

## 📁 Project Structure

```
llm_router/
├── config/                      # Model and policy configurations
│   ├── models.py
│   └── oracle_matrix.py
├── data/                        # Dataset generators
│   ├── benchmark_loader.py
│   ├── oracle_generator.py
│   ├── generate_dpo_calibration_data.py
│   ├── generate_sft_calibration_data.py
│   └── slm_baseline/            # Generated SLM baseline data
├── evaluation/                  # Judging and evaluation
│   ├── judge_slm_answers.py
│   └── judge_slm_answers_async.py
├── scripts/                     # Training and evaluation scripts
│   ├── modal_slm_baseline.py            # Generate SLM answers
│   ├── modal_dpo_training.py            # DPO calibration training
│   ├── modal_implicit_calibration_training.py  # Implicit training
│   ├── modal_sft_calibration_training.py       # SFT training
│   ├── modal_rl_calibration_vllm.py     # REINFORCE RL training
│   ├── modal_grpo_calibration_vllm.py   # GRPO RL training
│   ├── modal_evaluate_calibration.py    # Evaluate calibration
│   ├── modal_evaluate_rl_calibration.py # Evaluate RL models
│   └── modal_train_router.py            # Train router (original goal)
├── models/                      # Model clients
│   ├── gemini_client.py
│   └── vllm_client.py
├── CALIBRATION_RESEARCH.md      # Detailed research findings
└── README.md
```

---

## 📊 Key Metrics

### Calibration Metrics

- **RMS Calibration Error**: Root mean square error between predicted confidence and actual accuracy (lower is better)
- **Accuracy**: Percentage of correct answers
- **Average Confidence**: Mean confidence across all predictions
- **Calibration Curve**: Confidence vs accuracy binned plot

### Router Metrics (Future)

- **Routing Accuracy**: Percentage of queries routed to optimal model
- **Cost Savings**: Percentage reduction in API costs
- **Quality Maintenance**: Accuracy on complex queries

---

## 🔧 Configuration

### Models Used

**Small Language Models (for routing/calibration):**
- `google/gemma-2-2b-it` (primary)
- `google/gemma-2-9b-it`
- `Qwen/Qwen2.5-1.5B-Instruct`

**Target LLMs (for routing):**
- **Medium**: `gemini-2.5-flash`
- **Large**: `gemini-2.5-pro`

**Judge Model:**
- `gemini-2.5-pro`

### Training Configuration

```python
# LoRA Configuration
lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
}

# Training Hyperparameters
training_args = {
    "learning_rate": 1e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
}
```

---

## 📈 Current Research Findings

See [CALIBRATION_RESEARCH.md](CALIBRATION_RESEARCH.md) for detailed findings.

**Summary of Calibration Experiments:**

| Approach | RMS Calibration Error | Accuracy | Notes |
|----------|----------------------|----------|-------|
| **Baseline** (uncalibrated) | 0.3888 | 19.4% | Poor calibration |
| **DPO v1** | 0.3888 | 19.4% | No improvement (bug in data) |
| **DPO v2** | 0.3060 | 19.4% | 21% improvement, but flat 0.5 confidence |
| **Implicit** (buggy) | 0.1940 | 19.4% | All 0.0 confidence (evaluation bug) |
| **Implicit** (fixed) | Running... | Running... | Bug fixed, re-evaluating |
| **Standard SFT** | Training... | Training... | In progress |

**Key Bugs Discovered and Fixed:**
1. DPO v1 data had swapped chosen/rejected pairs
2. Implicit evaluation computed logprobs of INPUT instead of OUTPUT tokens
3. SFT data generation wrote literal `'\n'` instead of newlines

---

## 📊 Oracle Logic (9-Cell Matrix)

| Medium Model | Large Model | Policy           | Rationale                     |
|--------------|-------------|------------------|-------------------------------|
| Correct      | Correct     | Standard_Query   | Both correct → optimize cost  |
| Correct      | Incorrect   | Standard_Query   | Medium correct → use medium   |
| Correct      | IDK         | Standard_Query   | Medium correct is best        |
| Incorrect    | Correct     | Complex_Query    | Classic escalation            |
| Incorrect    | Incorrect   | Ambiguous_Query  | Both failed → don't escalate  |
| Incorrect    | IDK         | Complex_Query    | IDK safer than incorrect      |
| IDK          | Correct     | Complex_Query    | Large knows answer            |
| IDK          | Incorrect   | Ambiguous_Query  | IDK was safer                 |
| IDK          | IDK         | Ambiguous_Query  | Both failed → don't escalate  |

---

## 🎓 Key Innovations

1. **Decoupled Logic**: Policies separate from models
2. **Blind Judging**: Prevents judge self-preference
3. **Query Analysis**: Prevents data leakage in reasoning
4. **Multiple Calibration Approaches**: DPO, implicit, SFT
5. **Comprehensive Evaluation**: RMS error, calibration curves

---

## 🤝 Contributing

This is a research implementation. Key areas for improvement:

- Additional benchmarks (MMLU, TruthfulQA)
- Alternative calibration methods (Temperature scaling, Platt scaling)
- Multi-model routing (3+ models)
- Cost-accuracy tradeoff analysis
- Production optimization

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

Inspired by:
- Calibration research (Guo et al., 2017)
- DPO training (Rafailov et al., 2023)
- Google's Gemini model family
- TRL library for RLHF/DPO
- Modal for serverless compute

---

## 📧 Contact

For questions or collaborations, open an issue on GitHub.
