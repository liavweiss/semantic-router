#!/bin/bash
# Training script for RHEL system
# This replaces the Colab notebook workflow

set -e  # Exit on error

echo "🚀 Semantic Router Config Generation - Training on RHEL"
echo ""

# Configuration
BASE_MODEL_PATH="./base_model"  # Path to downloaded model
TRAIN_FILE="./train.jsonl"
VAL_FILE="./val.jsonl"
TEST_FILE="./test.jsonl"
OUTPUT_DIR="./models/config_gen_v1.0.0"
EPOCHS=5  # Increased from 3 to 5
BATCH_SIZE=1
GRADIENT_ACCUM=16
MAX_LENGTH=4096  # Increased back from 2048 to 4096
LEARNING_RATE=1e-5  # Decreased from 2e-5 to 1e-5
MAX_SAMPLES=50  # For evaluation

# Model selection: 1.5B or 3B
MODEL_SIZE="3B"  # Change to "1.5B" to use smaller model
if [ "$MODEL_SIZE" = "3B" ]; then
    MODEL_NAME="Qwen/Qwen2.5-Coder-3B-Instruct"
else
    MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Check GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not found. Training will use CPU (very slow!)"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Check Python Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Python version: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install transformers datasets peft accelerate torch pyyaml tqdm huggingface_hub

echo "✅ Dependencies installed"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Verify Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ ERROR: Base model not found at $BASE_MODEL_PATH"
    echo "   Please download the model first:"
    echo "   huggingface-cli download $MODEL_NAME --local-dir $BASE_MODEL_PATH"
    exit 1
fi
echo "✅ Base model found: $MODEL_SIZE ($MODEL_NAME)"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ ERROR: Training dataset not found at $TRAIN_FILE"
    exit 1
fi
echo "✅ Training dataset found"

if [ ! -f "$VAL_FILE" ]; then
    echo "❌ ERROR: Validation dataset not found at $VAL_FILE"
    exit 1
fi
echo "✅ Validation dataset found"

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ ERROR: Test dataset not found at $TEST_FILE"
    exit 1
fi
echo "✅ Test dataset found"

echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Evaluate Base Model (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if baseline already exists
BASELINE_FILE="evaluation_results_base_model_baseline.json"
if [ -f "$BASELINE_FILE" ]; then
    echo "✅ Baseline results already exist: $BASELINE_FILE"
    echo "📊 Skipping baseline evaluation (already done)"
    echo ""
    echo "To re-run baseline, delete the file:"
    echo "  rm $BASELINE_FILE"
    echo ""
else
    echo "⏱️  This will take ~5-10 minutes..."
    echo ""

    python3 train_config_gen_lora.py \
        --mode evaluate \
        --base-model "$BASE_MODEL_PATH" \
        --test-file "$TEST_FILE" \
        --max-samples $MAX_SAMPLES \
        --model-name "Base Model Baseline"

    echo ""
    echo "✅ Baseline evaluation completed!"
    echo "💾 Results saved to: $BASELINE_FILE"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Train Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  This will take ~30-60 minutes..."
echo ""

python3 train_config_gen_lora.py \
    --mode train \
    --base-model "$BASE_MODEL_PATH" \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --gradient-accumulation-steps $GRADIENT_ACCUM \
    --max-length $MAX_LENGTH

echo ""
echo "✅ Training completed!"
echo "💾 Model saved to: $OUTPUT_DIR"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Evaluate Trained Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  This will take ~5-10 minutes..."
echo ""

# Quick test
echo "🧪 Quick test with sample queries..."
python3 train_config_gen_lora.py \
    --mode test \
    --base-model "$BASE_MODEL_PATH" \
    --model-path "$OUTPUT_DIR"

echo ""

# Full evaluation
echo "📊 Full evaluation on test set..."
python3 train_config_gen_lora.py \
    --mode evaluate \
    --base-model "$BASE_MODEL_PATH" \
    --model-path "$OUTPUT_DIR" \
    --test-file "$TEST_FILE" \
    --max-samples $MAX_SAMPLES \
    --model-name "Trained Model"

echo ""
echo "✅ Evaluation completed!"
echo "💾 Results saved to: evaluation_results_trained_model.json"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Compare Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'EOF'
import json
import os

print("=" * 70)
print("📈 BASELINE vs TRAINED MODEL - COMPARISON")
print("=" * 70)
print("")

# Load results
baseline_files = [
    "evaluation_results_base_model_baseline.json",
]
trained_files = [
    "evaluation_results_trained_model.json",
]

baseline_results = None
trained_results = None

for bf in baseline_files:
    if os.path.exists(bf):
        with open(bf, 'r') as f:
            baseline_results = json.load(f)
        print(f"✅ Baseline results loaded")
        break

for tf in trained_files:
    if os.path.exists(tf):
        with open(tf, 'r') as f:
            trained_results = json.load(f)
        print(f"✅ Trained model results loaded")
        break

print("")

if baseline_results and trained_results:
    print("=" * 70)
    print("METRICS COMPARISON")
    print("=" * 70)
    print("")
    
    # YAML Accuracy
    baseline_acc = baseline_results['yaml_accuracy']
    trained_acc = trained_results['yaml_accuracy']
    acc_improvement = trained_acc - baseline_acc
    acc_change = "📈 IMPROVED" if acc_improvement > 0 else "📉 WORSE" if acc_improvement < 0 else "➡️  SAME"
    
    print(f"YAML Syntax Accuracy:")
    print(f"  Baseline:  {baseline_acc:.2f}%")
    print(f"  Trained:   {trained_acc:.2f}%")
    print(f"  Change:    {acc_improvement:+.2f}% {acc_change}")
    print("")
    
    # Error Rate
    baseline_err = baseline_results['error_rate']
    trained_err = trained_results['error_rate']
    err_improvement = baseline_err - trained_err
    err_change = "📈 IMPROVED" if err_improvement > 0 else "📉 WORSE" if err_improvement < 0 else "➡️  SAME"
    
    print(f"Error Rate:")
    print(f"  Baseline:  {baseline_err:.2f}%")
    print(f"  Trained:   {trained_err:.2f}%")
    print(f"  Change:    {err_improvement:+.2f}% {err_change}")
    print("")
    
    # Overall assessment
    print("=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    
    if acc_improvement > 5:
        print("✅ EXCELLENT: Training significantly improved the model!")
        print(f"   Accuracy improved by {acc_improvement:.2f} percentage points")
    elif acc_improvement > 0:
        print("✅ GOOD: Training improved the model")
        print(f"   Accuracy improved by {acc_improvement:.2f} percentage points")
    elif acc_improvement == 0:
        print("➡️  NEUTRAL: Training did not change performance")
    else:
        print("❌ WORSE: Training decreased performance")
        print(f"   Accuracy decreased by {abs(acc_improvement):.2f} percentage points")
    
    print("")
else:
    print("❌ Results files not found")
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL DONE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Results:"
echo "  - Baseline: evaluation_results_base_model_baseline.json"
echo "  - Trained:  evaluation_results_trained_model.json"
echo "  - Model:    $OUTPUT_DIR"
echo ""
echo "🎉 Training complete!"

