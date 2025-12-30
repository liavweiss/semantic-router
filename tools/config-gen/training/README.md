# Config Generation Model Training

This directory contains scripts and guides for training an LLM to generate semantic-router configurations.

## Overview

We fine-tune a base LLM (Llama 3.1 8B or Qwen 2.5 7B) using LoRA to generate YAML configurations from natural language descriptions.

## Quick Start

### Option 1: Google Colab (Recommended)

1. **Prepare files locally:**

   ```bash
   # 1. Download base model
   cd tools/config-gen/training
   python3 -c "
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model_name = 'meta-llama/Llama-3.1-8B-Instruct'
   print('Downloading base model...')
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer.save_pretrained('./base_model')
   model.save_pretrained('./base_model')
   print('✓ Saved to ./base_model')
   "
   
   # 2. Create ZIP
   zip -r base_model.zip base_model/
   
   # 3. Ensure dataset is ready
   cd ../dataset
   # Run pipeline to generate dataset
   ./scripts/run_pipeline.sh
   ```

2. **Upload to Google Drive:**
   - `base_model.zip` → `Projects/semantic-router/tools/config-gen/training/`
   - `train_config_gen_lora.py` → `Projects/semantic-router/tools/config-gen/training/`
   - `dataset/processed/train.jsonl` → `Projects/semantic-router/tools/config-gen/dataset/processed/`
   - `dataset/processed/val.jsonl` → `Projects/semantic-router/tools/config-gen/dataset/processed/`

3. **Open Colab notebook:**
   - Open `colab_training_notebook.ipynb` in Google Colab
   - Follow the cells step by step

### Option 2: Local Training

```bash
cd tools/config-gen/training

# Install dependencies
pip install transformers datasets peft accelerate torch pyyaml

# Train model
python train_config_gen_lora.py \
    --mode train \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --train-file ../dataset/processed/train.jsonl \
    --val-file ../dataset/processed/val.jsonl \
    --output-dir models/config_gen_llama3_8b_v1.0.0 \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5

# Test model
python train_config_gen_lora.py \
    --mode test \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --model-path models/config_gen_llama3_8b_v1.0.0
```

## Files

- **`train_config_gen_lora.py`**: Main training script
- **`colab_training_notebook.ipynb`**: Google Colab notebook for training
- **`README.md`**: This file

## Training Parameters

**Default settings:**

- Base model: `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-5
- LoRA rank: 16
- LoRA alpha: 32
- Max length: 4096

**Adjust based on:**

- GPU memory: Reduce batch size if OOM
- Dataset size: Increase epochs if small dataset
- Training loss: Adjust learning rate if not decreasing

## Expected Results

- **Training time**: ~15-30 minutes on Colab T4 GPU
- **Model size**: ~500MB (LoRA adapters only)
- **Validation**: >85% of generated configs should pass YAML syntax check

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch-size` to 2 or 1
- Reduce `--max-length` to 2048
- Use gradient checkpointing (add to training script)

### Slow Training

- Use GPU (Colab provides free T4)
- Reduce dataset size for testing
- Use smaller base model

### Poor Generation Quality

- Increase epochs (try 5)
- Increase LoRA rank (try 32)
- Check dataset quality
- Add more training examples

## Next Steps

After training:

1. Test model locally
2. Upload to HuggingFace (optional)
3. Integrate with API service
4. Deploy for config generation
