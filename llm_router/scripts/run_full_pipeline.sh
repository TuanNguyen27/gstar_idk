#!/bin/bash
# Full pipeline: Download → Generate Oracle → Prepare Training → Train → Evaluate

set -e  # Exit on error

echo "==================================="
echo "LLM Router Full Pipeline"
echo "==================================="

# Configuration
BENCHMARK_LIMIT=${BENCHMARK_LIMIT:-1000}  # Limit for training data generation
EVAL_LIMIT=${EVAL_LIMIT:-500}            # Limit for evaluation
MODEL_NAME=${MODEL_NAME:-"google/gemma-2b"}
SFT_EPOCHS=${SFT_EPOCHS:-3}
DPO_EPOCHS=${DPO_EPOCHS:-1}

# Check environment variables
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set"
    exit 1
fi

# Step 1: Download SimpleQA benchmark
echo ""
echo "Step 1: Downloading SimpleQA benchmark..."
python scripts/download_simpleqa.py \
    --output-dir ./data/benchmarks

SIMPLEQA_FILE="./data/benchmarks/simpleqa.jsonl"

# Step 2: Split into train/test sets
echo ""
echo "Step 2: Splitting dataset (80/20 train/test)..."
python scripts/split_dataset.py \
    --input-file $SIMPLEQA_FILE \
    --output-dir ./data/splits \
    --train-ratio 0.8 \
    --seed 42

TRAIN_FILE="./data/splits/simpleqa_train.jsonl"
TEST_FILE="./data/splits/simpleqa_test.jsonl"

# Step 3: Generate oracle dataset (TRAIN SET ONLY)
echo ""
echo "Step 3: Generating oracle dataset from TRAIN set (limit: $BENCHMARK_LIMIT)..."
python scripts/generate_oracle_dataset.py \
    --benchmark custom \
    --input-file $TRAIN_FILE \
    --output-dir ./data/generated \
    --limit $BENCHMARK_LIMIT \
    --gemini-api-key $GEMINI_API_KEY \
    --batch-size 10

# Step 4: Prepare training data
echo ""
echo "Step 4: Preparing SFT + DPO training data..."
python scripts/prepare_training_data.py \
    --oracle-file ./data/generated/oracle_dataset.jsonl \
    --output-dir ./data/training

# Step 5: Train router on Modal
echo ""
echo "Step 5: Training router (SFT + DPO)..."
echo "Model: $MODEL_NAME"
echo "SFT epochs: $SFT_EPOCHS, DPO epochs: $DPO_EPOCHS"

cd training
modal run modal_train_router.py \
    --model-name $MODEL_NAME \
    --sft-data-path ../data/training/sft_train.jsonl \
    --dpo-data-path ../data/training/dpo_train.jsonl \
    --sft-epochs $SFT_EPOCHS \
    --dpo-epochs $DPO_EPOCHS
cd ..

# Step 6: Download trained model from Modal
echo ""
echo "Step 6: Downloading trained model..."
modal volume get llm-router-models router_final ./router_checkpoints/router_final

# Step 7: Evaluate on TEST SET (no data leakage!)
echo ""
echo "Step 7: Evaluating router on TEST set (limit: $EVAL_LIMIT)..."
python scripts/evaluate_router.py \
    --model-path ./router_checkpoints/router_final \
    --test-file $TEST_FILE \
    --output-dir ./evaluation/results \
    --base-model $MODEL_NAME \
    --limit $EVAL_LIMIT \
    --gemini-api-key $GEMINI_API_KEY \
    --compare-baseline \
    --baseline-model gemini-2.5-flash

echo ""
echo "==================================="
echo "Pipeline Complete!"
echo "==================================="
echo ""
echo "Results:"
echo "  - Training data: ./data/training/"
echo "  - Trained model: ./router_checkpoints/router_final"
echo "  - Evaluation: ./evaluation/results/"
echo ""
echo "Next steps:"
echo "  1. Review evaluation metrics: ./evaluation/results/comparison.json"
echo "  2. Test router: python scripts/test_router.py --model-path ./router_checkpoints/router_final"
echo "  3. Deploy: modal deploy deployment/modal_inference.py"
