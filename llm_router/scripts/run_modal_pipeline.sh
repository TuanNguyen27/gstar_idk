#!/bin/bash
# Complete Modal-based pipeline

set -e

echo "==================================="
echo "LLM Router - Full Modal Pipeline"
echo "==================================="

# Configuration
LIMIT=${LIMIT:-40}  # Number of examples to generate
CONCURRENCY=${CONCURRENCY:-10}  # Parallel workers
MODEL_NAME=${MODEL_NAME:-"google/gemma-2b"}
SFT_EPOCHS=${SFT_EPOCHS:-3}
DPO_EPOCHS=${DPO_EPOCHS:-1}

# Check environment
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set"
    exit 1
fi

echo "Configuration:"
echo "  Examples: $LIMIT"
echo "  Concurrency: $CONCURRENCY"
echo "  Model: $MODEL_NAME"
echo ""

# Step 1: Generate oracle dataset on Modal (PARALLEL)
echo "Step 1: Generating oracle dataset on Modal..."
echo "Expected time: 3-5 minutes for $LIMIT examples"
echo ""

modal run scripts/modal_generate_oracle.py \
    --input-file ./data/splits/simpleqa_train.jsonl \
    --output-dir ./data/generated \
    --limit $LIMIT \
    --concurrency $CONCURRENCY

echo ""
echo "✓ Oracle generation complete!"

# Step 2: Prepare training data (LOCAL, fast)
echo ""
echo "Step 2: Preparing SFT + DPO training data..."

python3 scripts/prepare_training_data.py \
    --oracle-file ./data/generated/oracle_dataset.jsonl \
    --output-dir ./data/training

echo "✓ Training data prepared!"

# Step 3: Train router on Modal
echo ""
echo "Step 3: Training router on Modal..."
echo "Expected time: 2-3 hours"
echo ""

cd training
modal run modal_train_router.py \
    --model-name $MODEL_NAME \
    --sft-data-path ../data/training/sft_train.jsonl \
    --dpo-data-path ../data/training/dpo_train.jsonl \
    --sft-epochs $SFT_EPOCHS \
    --dpo-epochs $DPO_EPOCHS
cd ..

echo ""
echo "✓ Training complete!"

# Step 4: Download trained model
echo ""
echo "Step 4: Downloading trained model from Modal..."

modal volume get llm-router-models router_final ./router_checkpoints/router_final

echo "✓ Model downloaded!"

# Step 5: Evaluate on test set
echo ""
echo "Step 5: Evaluating router on test set..."

python3 scripts/evaluate_router.py \
    --model-path ./router_checkpoints/router_final \
    --test-file ./data/splits/simpleqa_test.jsonl \
    --output-dir ./evaluation/results \
    --base-model $MODEL_NAME \
    --gemini-api-key $GEMINI_API_KEY \
    --compare-baseline \
    --baseline-model gemini-2.5-flash

echo ""
echo "==================================="
echo "✓ Pipeline Complete!"
echo "==================================="
echo ""
echo "Results:"
echo "  - Oracle data: ./data/generated/oracle_dataset.jsonl"
echo "  - Training data: ./data/training/"
echo "  - Trained model: ./router_checkpoints/router_final"
echo "  - Evaluation: ./evaluation/results/comparison.json"
echo ""
echo "View metrics:"
echo "  cat ./evaluation/results/comparison.json | python3 -m json.tool"
echo ""
