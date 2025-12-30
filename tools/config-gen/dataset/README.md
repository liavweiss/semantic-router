# Semantic Router Configuration Dataset

This directory contains scripts and data for generating training datasets for LLM-based configuration generation.

## Overview

The dataset generation process uses multiple strategies to create a diverse set of configuration examples:

1. **Real Examples**: Collected from the codebase (~39 examples)
2. **AI-Generated Examples**: Directly generated synthetic examples (~300 examples)
3. **Template Variations**: Systematic variations from templates (~200 examples)
4. **Augmented Examples**: Modifications of existing configs (~94 examples)
5. **Section Examples**: Individual config sections (~78 examples)

**Current Dataset**: 711 total examples (validated and ready for training)

## Directory Structure

```
dataset/
├── raw/                    # Raw collected/generated examples
│   ├── real_examples.jsonl
│   ├── synthetic_ai_examples.jsonl
│   ├── template_examples.jsonl
│   ├── augmented_examples.jsonl
│   └── section_examples.jsonl
├── annotated/              # Manually annotated examples (future)
├── processed/              # Validated and split datasets
│   ├── validated.jsonl     # All validated examples (711)
│   ├── train.jsonl         # Training set (568 examples, 80%)
│   ├── val.jsonl           # Validation set (71 examples, 10%)
│   ├── test.jsonl          # Test set (72 examples, 10%)
│   ├── invalid.jsonl       # Invalid examples (for debugging)
│   └── statistics.json     # Dataset statistics
├── templates/              # Config templates
├── scripts/                 # Generation scripts
│   ├── collect_real.py
│   ├── generate_ai_examples.py
│   ├── generate_from_templates.py
│   ├── augment_existing.py
│   ├── generate_sections.py
│   ├── validate_and_filter.py
│   ├── combine_and_split.py
│   └── run_pipeline.sh
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd semantic-router/tools/config-gen/dataset
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run pipeline (no API keys needed - uses direct generation)
./scripts/run_pipeline.sh
```

### 3. Run Individual Steps

```bash
# Step 1: Collect real examples
python scripts/collect_real.py --base-dir ../../.. --output raw/real_examples.jsonl

# Step 2: Generate AI examples (direct generation, no API needed)
python scripts/generate_ai_examples.py \
    --output raw/synthetic_ai_examples.jsonl \
    --num-examples 300

# Step 3: Generate template variations
python scripts/generate_from_templates.py \
    --max-variations 200 \
    --output raw/template_examples.jsonl

# Step 4: Generate augmentations
python scripts/augment_existing.py \
    --input raw/real_examples.jsonl \
    --num-augmentations 100 \
    --output raw/augmented_examples.jsonl

# Step 5: Generate section examples
python scripts/generate_sections.py \
    --input raw/real_examples.jsonl \
    --output raw/section_examples.jsonl

# Step 6: Validate and filter
python scripts/validate_and_filter.py \
    --input raw/*.jsonl \
    --output processed/validated.jsonl

# Step 7: Combine and split
python scripts/combine_and_split.py \
    --input processed/validated.jsonl \
    --output-dir processed \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --deduplicate \
    --training-format
```

## Scripts Documentation

### collect_real.py
Collects real configuration files from the codebase.

**Usage:**

```bash
python scripts/collect_real.py --base-dir ../../.. --output raw/real_examples.jsonl
```

### generate_ai_examples.py
Generates synthetic examples directly (no API needed). Creates diverse configurations with various models, deployment contexts, and feature combinations.

**Usage:**

```bash
python scripts/generate_ai_examples.py \
    --output raw/synthetic_ai_examples.jsonl \
    --num-examples 300
```

**Options:**

- `--output`: Output JSONL file path (default: `raw/synthetic_ai_examples.jsonl`)
- `--num-examples`: Number of examples to generate (default: 150)

**Features:**

- Generates configs with different models (qwen3, llama3, phi4, mistral, deepseek, gpt-oss)
- Covers 8 deployment contexts (quickstart, ai-gateway, istio, kubernetes, etc.)
- Varies complexity (low, medium, high)
- Includes different feature combinations (cache, PII detection, tools, observability)

### generate_from_templates.py
Generates systematic variations from config templates.

**Usage:**

```bash
python scripts/generate_from_templates.py \
    --max-variations 200 \
    --output raw/template_examples.jsonl
```

### augment_existing.py
Creates variations by modifying existing configs.

**Usage:**

```bash
python scripts/augment_existing.py \
    --input raw/real_examples.jsonl \
    --num-augmentations 100 \
    --output raw/augmented_examples.jsonl
```

### generate_sections.py
Extracts individual config sections for partial generation examples.

**Usage:**

```bash
python scripts/generate_sections.py \
    --input raw/real_examples.jsonl \
    --output raw/section_examples.jsonl
```

### validate_and_filter.py
Validates YAML syntax and optionally semantic validation.

**Usage:**

```bash
python scripts/validate_and_filter.py \
    --input raw/*.jsonl \
    --output processed/validated.jsonl \
    --errors-output processed/invalid.jsonl
```

### combine_and_split.py
Combines all examples and splits into train/val/test sets.

**Usage:**

```bash
python scripts/combine_and_split.py \
    --input processed/validated.jsonl \
    --output-dir processed \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --deduplicate \
    --training-format
```

## Data Format

### Input Format (JSONL)
Each line is a JSON object:

```json
{
  "id": "real_config_main_1234",
  "source_file": "config/config.yaml",
  "deployment_context": "quickstart",
  "intent": "Complete configuration for local development",
  "full_config": "decisions:\n  - name: ...",
  "source": "real",
  "key_features": ["semantic_cache", "routing_decisions"]
}
```

### Training Format (JSONL)
Each line is a JSON object:

```json
{
  "instruction": "Generate semantic-router configuration that routes business queries to qwen3 model",
  "input": "",
  "output": "decisions:\n  - name: \"business_decision\"\n    ..."
}
```

## Current Dataset Statistics

**As of latest generation:**

- **Total examples**: 711 (all validated)
- **Training set**: 568 examples (80%)
- **Validation set**: 71 examples (10%)
- **Test set**: 72 examples (10%)

**Breakdown by source:**

- Real examples: 39
- AI-generated: 300
- Template variations: 200
- Augmented: 94
- Section examples: 78

**Coverage:**

- 8 deployment contexts
- 7 different models
- 3 complexity levels
- Multiple feature combinations

## Costs

**All generation methods:**

- **Free** - No API costs (uses direct generation)
- **Time**: ~30-60 minutes for full pipeline
- **No API keys required**

## Next Steps

After generating the dataset:

1. Review statistics in `processed/statistics.json`
2. Manually review sample examples for quality
3. Use `processed/train.jsonl`, `processed/val.jsonl`, `processed/test.jsonl` for training

## Training Readiness

**Is 711 examples enough for training?**

✅ **Yes, for POC/Medium models (7-8B):**

- 700+ examples is sufficient for fine-tuning medium-sized models
- Good coverage of deployment contexts and features
- Balanced complexity distribution

💡 **For production/larger models:**

- Consider generating more examples (1000-2000+)
- Focus on underrepresented deployment contexts
- Add more high-complexity examples

**Recommended model sizes:**

- Small (1-3B): 100-500 examples ✅
- Medium (7-8B): 500-2000 examples ✅ (we have 711)
- Large (13B+): 2000+ examples (consider more)

## Troubleshooting

### Validation Failures

- Check `processed/invalid.jsonl` for error details
- Common issues: YAML syntax errors, missing required fields

### Low Pass Rate

- Review validation errors
- Adjust generation prompts/parameters
- Add more seed examples

## References

- See `data-augmentation-strategy.md` for detailed strategy
- See `plan-training.md` for training plan
