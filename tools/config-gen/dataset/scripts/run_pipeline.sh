#!/bin/bash
# Data Generation Pipeline
# This script runs the complete data generation pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")"
BASE_DIR="$(cd "$DATASET_DIR/../../.." && pwd)"

echo "=== Data Generation Pipeline ==="
echo "Base directory: $BASE_DIR"
echo "Dataset directory: $DATASET_DIR"
echo ""

# Configuration
NUM_AI_EXAMPLES=${NUM_AI_EXAMPLES:-300}
NUM_TEMPLATE_VARIATIONS=${NUM_TEMPLATE_VARIATIONS:-200}
NUM_AUGMENTATIONS=${NUM_AUGMENTATIONS:-100}

# Step 1: Collect real examples
echo "Step 1: Collecting real config examples..."
python "$SCRIPT_DIR/collect_real.py" \
    --base-dir "$BASE_DIR" \
    --output "$DATASET_DIR/raw/real_examples.jsonl"

if [ ! -f "$DATASET_DIR/raw/real_examples.jsonl" ]; then
    echo "Error: Failed to collect real examples"
    exit 1
fi

REAL_COUNT=$(wc -l < "$DATASET_DIR/raw/real_examples.jsonl")
echo "Collected $REAL_COUNT real examples"
echo ""

# Step 2: Generate AI examples (direct generation, no API needed)
echo "Step 2: Generating AI examples..."
NUM_AI_EXAMPLES=${NUM_AI_EXAMPLES:-300}
python "$SCRIPT_DIR/generate_ai_examples.py" \
    --output "$DATASET_DIR/raw/synthetic_ai_examples.jsonl" \
    --num-examples "$NUM_AI_EXAMPLES"
echo ""

# Step 3: Generate template variations
echo "Step 3: Generating template variations..."
python "$SCRIPT_DIR/generate_from_templates.py" \
    --max-variations "$NUM_TEMPLATE_VARIATIONS" \
    --output "$DATASET_DIR/raw/template_examples.jsonl"
echo ""

# Step 4: Generate augmentations
echo "Step 4: Generating augmented examples..."
python "$SCRIPT_DIR/augment_existing.py" \
    --input "$DATASET_DIR/raw/real_examples.jsonl" \
    --num-augmentations "$NUM_AUGMENTATIONS" \
    --output "$DATASET_DIR/raw/augmented_examples.jsonl"
echo ""

# Step 5: Generate section examples
echo "Step 5: Generating section-level examples..."
python "$SCRIPT_DIR/generate_sections.py" \
    --input "$DATASET_DIR/raw/real_examples.jsonl" \
    --output "$DATASET_DIR/raw/section_examples.jsonl"
echo ""

# Step 6: Validate and filter
echo "Step 6: Validating and filtering examples..."
INPUT_FILES=(
    "$DATASET_DIR/raw/real_examples.jsonl"
    "$DATASET_DIR/raw/synthetic_ai_examples.jsonl"
    "$DATASET_DIR/raw/template_examples.jsonl"
    "$DATASET_DIR/raw/augmented_examples.jsonl"
    "$DATASET_DIR/raw/section_examples.jsonl"
)

# Only include files that exist
EXISTING_FILES=()
for file in "${INPUT_FILES[@]}"; do
    if [ -f "$file" ]; then
        EXISTING_FILES+=("$file")
    fi
done

if [ ${#EXISTING_FILES[@]} -eq 0 ]; then
    echo "Error: No input files found for validation"
    exit 1
fi

python "$SCRIPT_DIR/validate_and_filter.py" \
    --input "${EXISTING_FILES[@]}" \
    --output "$DATASET_DIR/processed/validated.jsonl" \
    --errors-output "$DATASET_DIR/processed/invalid.jsonl"
echo ""

# Step 7: Combine and split
echo "Step 7: Combining and splitting into train/val/test..."
python "$SCRIPT_DIR/combine_and_split.py" \
    --input "$DATASET_DIR/processed/validated.jsonl" \
    --output-dir "$DATASET_DIR/processed" \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --deduplicate \
    --training-format
echo ""

echo "=== Pipeline Complete ==="
echo "Final dataset location: $DATASET_DIR/processed/"
echo "  - train.jsonl"
echo "  - val.jsonl"
echo "  - test.jsonl"
echo "  - statistics.json"

