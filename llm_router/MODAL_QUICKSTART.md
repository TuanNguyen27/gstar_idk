# Modal Quick Start Guide

Complete guide to running the LLM Router pipeline on Modal for 10x faster execution.

## Prerequisites

```bash
# Install Modal
pip install modal

# Set Gemini API key
export GEMINI_API_KEY="your-key-here"
```

## One-Time Setup

```bash
# Run setup script (interactive)
./scripts/setup_modal.sh
```

This will:
1. Install Modal CLI
2. Authenticate with Modal (opens browser)
3. Create secrets (gemini-secret, huggingface-secret)
4. Test your Modal setup

## Running the Full Pipeline

### Option 1: One Command (Recommended)

```bash
# Run everything on Modal
./scripts/run_modal_pipeline.sh
```

**Default configuration:**
- 40 training examples (30 seconds on Modal vs 30 minutes locally!)
- 10 parallel workers
- Gemma-2B router model
- Full SFT + DPO training

**Custom configuration:**
```bash
# Generate more examples
LIMIT=100 ./scripts/run_modal_pipeline.sh

# Use different model
MODEL_NAME="Qwen/Qwen2.5-1.5B" ./scripts/run_modal_pipeline.sh

# More parallel workers
CONCURRENCY=20 LIMIT=200 ./scripts/run_modal_pipeline.sh
```

### Option 2: Step-by-Step

#### Step 1: Generate Oracle Dataset (Modal - FAST!)

```bash
# 40 examples in ~3-5 minutes (vs 30 min locally)
modal run scripts/modal_generate_oracle.py \
  --input-file ./data/splits/simpleqa_train.jsonl \
  --output-dir ./data/generated \
  --limit 40 \
  --concurrency 10
```

**What happens:**
- 10 examples processed in parallel
- Automatic retry on failures
- Better safety settings for factual QA
- Progress updates in real-time

**Expected output:**
```
=== Modal Oracle Dataset Generation ===
Input: ./data/splits/simpleqa_train.jsonl
Output: ./data/generated
Limit: 40
Concurrency: 10

Loaded 40 examples

Step 1: Generating model answers in parallel...
  Progress: 10/40
  Progress: 20/40
  Progress: 30/40
  Progress: 40/40
✓ Generated 40 model answer pairs

Step 2: Judging answers and applying oracle matrix...
  Progress: 8 valid examples
  Progress: 16 valid examples
  Progress: 24 valid examples
  Progress: 32 valid examples
✓ Generated 32 oracle examples

=== Generation Complete ===
Total examples: 32
Output: data/generated/oracle_dataset.jsonl

Policy Distribution:
  Standard_Query: 18 (56.2%)
  Complex_Query: 10 (31.2%)
  Ambiguous_Query: 4 (12.5%)
```

#### Step 2: Prepare Training Data (Local - Instant)

```bash
python3 scripts/prepare_training_data.py \
  --oracle-file ./data/generated/oracle_dataset.jsonl \
  --output-dir ./data/training
```

**Output:**
- `data/training/sft_train.jsonl` - For supervised fine-tuning
- `data/training/dpo_train.jsonl` - For preference optimization

#### Step 3: Train Router (Modal - GPU)

```bash
cd training
modal run modal_train_router.py \
  --model-name google/gemma-2b \
  --sft-data-path ../data/training/sft_train.jsonl \
  --dpo-data-path ../data/training/dpo_train.jsonl \
  --sft-epochs 3 \
  --dpo-epochs 1
```

**Expected time:**
- SFT: 1-2 hours (A100 GPU)
- DPO: 30-60 minutes
- **Total: ~2-3 hours**

#### Step 4: Download Trained Model

```bash
modal volume get llm-router-models router_final ./router_checkpoints/router_final
```

#### Step 5: Evaluate

```bash
python3 scripts/evaluate_router.py \
  --model-path ./router_checkpoints/router_final \
  --test-file ./data/splits/simpleqa_test.jsonl \
  --output-dir ./evaluation/results \
  --gemini-api-key $GEMINI_API_KEY \
  --compare-baseline
```

## Timing Comparison: Local vs Modal

| Task | Local | Modal | Speedup |
|------|-------|-------|---------|
| **Oracle Gen (40 examples)** | 30 min | 3-5 min | **6-10x** |
| **Training (SFT + DPO)** | N/A (needs GPU) | 2-3 hours | GPU access |
| **Total Pipeline** | Hours (if GPU) | **2-3 hours** | Faster + reliable |

## Cost Estimates

### Oracle Generation
- Gemini API: ~$0.20 for 40 examples
- Modal compute: Free tier sufficient

### Training
- Modal GPU (A100): ~$1.50/hour
- Total training: ~$3-5 for full pipeline

### Total Cost
- **$5-10 for complete pipeline** (data generation + training + evaluation)

## Monitoring

### Check Modal Dashboard
```bash
modal app logs llm-router-oracle-generation
modal app logs llm-router-training
```

### View in Browser
- Dashboard: https://modal.com/apps
- Real-time logs and metrics

### Check Volumes
```bash
# List volumes
modal volume list

# View contents
modal volume ls llm-router-data
modal volume ls llm-router-models
```

## Troubleshooting

### Authentication Issues
```bash
# Re-authenticate
modal token new

# Check current token
modal token list
```

### Secret Not Found
```bash
# Recreate secrets
modal secret create gemini-secret GEMINI_API_KEY="your-key"
modal secret create huggingface-secret HF_TOKEN="your-token"

# List secrets
modal secret list
```

### Out of Memory (Training)
```bash
# Reduce batch size
# Edit training/modal_train_router.py:
# batch_size=2 (instead of 4)
```

### API Rate Limits
```bash
# Reduce concurrency
modal run scripts/modal_generate_oracle.py --concurrency 5
```

## Advanced Usage

### Custom Safety Settings

Edit `scripts/modal_generate_oracle.py`:
```python
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    # Adjust as needed
]
```

### Different Router Models

```bash
# Smaller model (faster, cheaper)
modal run modal_train_router.py --model-name Qwen/Qwen2.5-1.5B

# Larger model (better quality)
modal run modal_train_router.py --model-name google/gemma-7b
```

### Parallel Generation at Scale

```bash
# 500 examples in ~15-20 minutes
modal run scripts/modal_generate_oracle.py \
  --limit 500 \
  --concurrency 20
```

## Best Practices

1. **Start Small**: Test with 10-20 examples first
2. **Monitor Costs**: Check Modal dashboard regularly
3. **Save Checkpoints**: Training saves every 500 steps
4. **Version Control**: Tag successful runs
5. **Test Locally First**: Validate scripts before Modal deployment

## Next Steps

After completing the pipeline:

1. **Test the Router**:
   ```bash
   python3 scripts/test_router.py \
     --model-path ./router_checkpoints/router_final \
     --query "What is the wingspan of a Pteranodon?"
   ```

2. **Deploy to Production**:
   ```bash
   modal deploy deployment/modal_inference.py
   ```

3. **Iterate**:
   - Analyze failure cases
   - Generate more training data for weak areas
   - Retrain and evaluate

## Support

- Modal Docs: https://modal.com/docs
- LLM Router Issues: GitHub repository
- Modal Slack: https://modal.com/slack

## Pro Tips

💡 **Use Modal for everything API-heavy or GPU-intensive**
💡 **Keep local scripts for quick data processing**
💡 **Monitor your Gemini API quota**
💡 **Save intermediate results to volumes**
💡 **Use spot instances for cost savings** (add `gpu=modal.gpu.A100(count=1, memory=40)` with spot=True)
