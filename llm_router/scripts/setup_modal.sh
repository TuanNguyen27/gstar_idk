#!/bin/bash
# Setup Modal for LLM Router project

set -e

echo "==================================="
echo "Modal Setup for LLM Router"
echo "==================================="

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "Installing Modal..."
    pip install modal
else
    echo "✓ Modal already installed"
fi

# Authenticate with Modal
echo ""
echo "Step 1: Modal Authentication"
echo "If you haven't authenticated yet, run: modal token new"
echo "This will open a browser to authenticate."
read -p "Have you authenticated with Modal? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Running: modal token new"
    modal token new
fi

# Create Modal secrets
echo ""
echo "Step 2: Creating Modal Secrets"

# Check if Gemini API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set"
    read -p "Enter your Gemini API key: " GEMINI_API_KEY
    export GEMINI_API_KEY
fi

# Create gemini-secret
echo "Creating gemini-secret..."
modal secret create gemini-secret GEMINI_API_KEY="$GEMINI_API_KEY" 2>/dev/null || \
    echo "Note: gemini-secret may already exist (this is OK)"

# Create huggingface-secret (for model downloads)
echo ""
echo "Step 3: HuggingFace Token (Optional, for model training)"
read -p "Do you have a HuggingFace token? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your HuggingFace token: " HF_TOKEN
    modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN" 2>/dev/null || \
        echo "Note: huggingface-secret may already exist (this is OK)"
else
    echo "Skipping HuggingFace token (you can add it later)"
fi

# Test Modal setup
echo ""
echo "Step 4: Testing Modal Setup"
echo "Running a simple test..."

cat > /tmp/modal_test.py <<'EOF'
import modal

app = modal.App("test")

@app.function()
def hello():
    return "Modal is working!"

@app.local_entrypoint()
def main():
    result = hello.remote()
    print(f"✓ {result}")
EOF

python3 -m modal run /tmp/modal_test.py

echo ""
echo "==================================="
echo "✓ Modal Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  1. Generate oracle dataset:"
echo "     modal run scripts/modal_generate_oracle.py --limit 40"
echo ""
echo "  2. Prepare training data:"
echo "     python3 scripts/prepare_training_data.py --oracle-file ./data/generated/oracle_dataset.jsonl"
echo ""
echo "  3. Train router:"
echo "     modal run training/modal_train_router.py"
echo ""
