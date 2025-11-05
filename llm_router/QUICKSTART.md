# LLM Router - Quick Start Guide

## Overview

This project implements a **performance-aligned LLM router** that intelligently routes queries to the optimal model (Gemini Flash vs Pro) based on complexity, reducing costs while maintaining accuracy.

## Key Features

✅ **Oracle-based training**: Ground truth from comparative model performance
✅ **Bias mitigation**: Blind judging prevents self-preference
✅ **No data leakage**: Train/test split + query-only analysis
✅ **Flexible routing**: Dynamic policy mapping (not rigid JSON)
✅ **Two-stage training**: SFT + DPO on Modal
✅ **SimpleQA evaluation**: Standard benchmark from OpenAI

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export GEMINI_API_KEY="your-gemini-key"
export MODAL_TOKEN_ID="your-modal-token-id"
export MODAL_TOKEN_SECRET="your-modal-token-secret"
```

## One-Command Pipeline

```bash
# Run the full pipeline (download → generate → train → evaluate)
./scripts/run_full_pipeline.sh
```

This will:
1. Download SimpleQA benchmark
2. Split into train (80%) / test (20%)
3. Generate oracle dataset from **TRAIN SET ONLY**
4. Prepare SFT + DPO training data
5. Train router on Modal (SFT + DPO)
6. Download trained model
7. Evaluate on **TEST SET** (no leakage!)

## Step-by-Step Usage

### 1. Download SimpleQA

```bash
python scripts/download_simpleqa.py --output-dir ./data/benchmarks
```

### 2. Split Dataset

```bash
python scripts/split_dataset.py \
  --input-file ./data/benchmarks/simpleqa.jsonl \
  --output-dir ./data/splits \
  --train-ratio 0.8
```

### 3. Generate Oracle Dataset

```bash
python scripts/generate_oracle_dataset.py \
  --benchmark custom \
  --input-file ./data/splits/simpleqa_train.jsonl \
  --output-dir ./data/generated \
  --limit 1000 \
  --gemini-api-key $GEMINI_API_KEY
```

**What this does:**
- Generates answers from both Flash and Pro models
- Judge answers blindly (prevents bias)
- Applies 9-cell oracle matrix to label policies
- Analyzes query characteristics (no data leakage)

### 4. Prepare Training Data

```bash
python scripts/prepare_training_data.py \
  --oracle-file ./data/generated/oracle_dataset.jsonl \
  --output-dir ./data/training
```

Creates:
- `sft_train.jsonl` - For supervised fine-tuning
- `dpo_train.jsonl` - For preference optimization

### 5. Train Router

```bash
cd training
modal run modal_train_router.py \
  --model-name google/gemma-2b \
  --sft-data-path ../data/training/sft_train.jsonl \
  --dpo-data-path ../data/training/dpo_train.jsonl \
  --sft-epochs 3 \
  --dpo-epochs 1
```

**Training specs:**
- GPU: A100 (40GB)
- Method: QLoRA (4-bit quantization)
- Stage 1: SFT (3 epochs, ~1-2 hours)
- Stage 2: DPO (1 epoch, ~30-60 minutes)

### 6. Download Model

```bash
modal volume get llm-router-models router_final ./router_checkpoints/router_final
```

### 7. Evaluate

```bash
python scripts/evaluate_router.py \
  --model-path ./router_checkpoints/router_final \
  --test-file ./data/splits/simpleqa_test.jsonl \
  --output-dir ./evaluation/results \
  --gemini-api-key $GEMINI_API_KEY \
  --compare-baseline
```

**Evaluation metrics:**
- Accuracy (vs baseline)
- Cost savings
- Policy distribution
- Per-policy accuracy

## Testing the Router

### Quick Test

```bash
python scripts/test_router.py \
  --model-path ./router_checkpoints/router_final \
  --query "What is the wingspan of a Pteranodon?"
```

### End-to-End Test (Routing + Answer)

```bash
python scripts/test_router.py \
  --model-path ./router_checkpoints/router_final \
  --query "What is 2+2?" \
  --end-to-end \
  --gemini-api-key $GEMINI_API_KEY
```

### Batch Testing

```bash
# Create test_queries.txt with one query per line
python scripts/test_router.py \
  --model-path ./router_checkpoints/router_final \
  --test-file test_queries.txt
```

## Flexible Policy Mapping

The router outputs policies (`Standard_Query`, `Complex_Query`, `Ambiguous_Query`), which are mapped to models via flexible logic:

### Static Mapping (Default)

```json
{
  "Standard_Query": "gemini-2.5-flash",
  "Complex_Query": "gemini-2.5-pro",
  "Ambiguous_Query": "gemini-2.5-flash"
}
```

### Dynamic Mapping (Recommended)

```python
from deployment.flexible_policy_map import get_hybrid_mapper

# Cost-aware + keyword-based + length-based + math detection
mapper = get_hybrid_mapper(
    cost_budget=10.0,  # $10 budget
    current_spend=2.5,  # $2.50 spent so far
)

model = mapper.get_model(query="Write Python code for...", policy="Standard_Query")
# Returns: "gemini-2.5-pro" (keyword override: "code")
```

**Available mappers:**
- `get_cost_aware_mapper()` - Downgrade when near budget
- `get_keyword_mapper()` - Override based on keywords (code, math, etc.)
- `get_hybrid_mapper()` - All rules combined (recommended)

## Configuration

### Environment Variables

```bash
# Required
export GEMINI_API_KEY="your-key"

# Optional (for Modal)
export MODAL_TOKEN_ID="your-token-id"
export MODAL_TOKEN_SECRET="your-token-secret"

# Pipeline tuning
export BENCHMARK_LIMIT=1000  # Training examples
export EVAL_LIMIT=500        # Test examples
export MODEL_NAME="google/gemma-2b"
export SFT_EPOCHS=3
export DPO_EPOCHS=1
```

### Model Selection

Change router base model in training:

```bash
# Gemma-2B (default)
--model-name google/gemma-2b

# Qwen 1.5B (smaller, faster)
--model-name Qwen/Qwen2.5-1.5B

# Phi-2 (2.7B, alternative)
--model-name microsoft/phi-2
```

## Expected Results

Based on research design:

| Metric | Expected Value |
|--------|---------------|
| Accuracy vs Flash-only | Similar or +2-5% |
| Accuracy vs Pro-only | -2-5% (acceptable tradeoff) |
| Cost savings | 40-60% |
| Flash usage | 60-70% of queries |
| Pro usage | 25-35% of queries |
| Ambiguous | 5-10% of queries |

## Project Structure

```
llm_router/
├── config/              # Models, policies, oracle matrix
├── data/                # Dataset loaders, oracle generator
├── training/            # SFT + DPO training (Modal)
├── deployment/          # Inference, flexible policy mapping
├── evaluation/          # SimpleQA evaluator
├── scripts/             # CLI tools
│   ├── download_simpleqa.py
│   ├── split_dataset.py
│   ├── generate_oracle_dataset.py
│   ├── prepare_training_data.py
│   ├── evaluate_router.py
│   ├── test_router.py
│   └── run_full_pipeline.sh
└── README.md
```

## Troubleshooting

### Modal Authentication

```bash
modal token new
# Follow prompts to authenticate
```

### Gemini API Rate Limits

If you hit rate limits during oracle generation:

```bash
# Reduce batch size and add delays
python scripts/generate_oracle_dataset.py \
  --batch-size 5 \
  --limit 100  # Start small
```

### Out of Memory (Training)

```bash
# Use smaller batch size
# Edit training/modal_train_router.py:
# batch_size=2 (instead of 4)
```

### Model Download Issues

```bash
# Set HuggingFace token
export HF_TOKEN="your-token"

# Add to Modal secrets
modal secret create huggingface-secret HF_TOKEN=$HF_TOKEN
```

## Next Steps

1. **Analyze Results**: Check `./evaluation/results/comparison.json`
2. **Test Deployment**: Deploy to Modal with `modal deploy deployment/modal_inference.py`
3. **Iterate**: Generate more training data if accuracy is low
4. **Optimize**: Tune flexible policy mapping rules for your use case

## Resources

- [SimpleQA Benchmark](https://github.com/openai/simple-evals)
- [Modal Documentation](https://modal.com/docs)
- [Gemini API](https://ai.google.dev/gemini-api/docs)
- [TRL Library](https://huggingface.co/docs/trl)

## Support

For issues or questions:
- Check `README.md` for detailed architecture
- Review evaluation metrics in `./evaluation/results/`
- Open an issue on GitHub
