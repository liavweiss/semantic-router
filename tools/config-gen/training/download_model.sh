#!/bin/bash
# Model Download Helper Script
# Usage: bash download_model.sh [qwen|llama]

set -e

MODEL_TYPE=${1:-qwen}

echo "🚀 Model Download Helper"
echo "========================"
echo ""

if [ "$MODEL_TYPE" = "qwen" ]; then
    echo "📥 Downloading Qwen 2.5 7B Instruct..."
    echo "   (No access required - this will work immediately)"
    echo ""
    
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'📥 Downloading {model_name}...')
print('⏱️  This may take 10-20 minutes depending on your connection...')
print('')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained('./base_model')
model.save_pretrained('./base_model')

print('')
print('✅ Model downloaded successfully!')
print('📁 Saved to: ./base_model')
"
    
    MODEL_NAME="Qwen 2.5 7B"
    
elif [ "$MODEL_TYPE" = "llama" ]; then
    echo "📥 Downloading Llama 3.1 8B Instruct..."
    echo "   (Requires HuggingFace access - make sure you have approval!)"
    echo ""
    
    echo "🔐 Please login to HuggingFace..."
    python3 -c "
from huggingface_hub import login
import getpass

print('Please enter your HuggingFace token:')
print('Get it from: https://huggingface.co/settings/tokens')
token = getpass.getpass('Token: ')
login(token=token)
print('✅ Logged in to HuggingFace')
"
    
    echo ""
    echo "📥 Downloading model..."
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
print(f'📥 Downloading {model_name}...')
print('⏱️  This may take 15-30 minutes depending on your connection...')
print('')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained('./base_model')
model.save_pretrained('./base_model')

print('')
print('✅ Model downloaded successfully!')
print('📁 Saved to: ./base_model')
"
    
    MODEL_NAME="Llama 3.1 8B"
    
else
    echo "❌ Error: Invalid model type. Use 'qwen' or 'llama'"
    echo ""
    echo "Usage: bash download_model.sh [qwen|llama]"
    exit 1
fi

echo ""
echo "📦 Creating ZIP archive..."
if [ -d "base_model" ]; then
    zip -r base_model.zip base_model/
    
    echo ""
    echo "✅ ZIP created successfully!"
    echo ""
    echo "📊 Summary:"
    echo "  Model: $MODEL_NAME"
    echo "  Directory: ./base_model"
    echo "  ZIP file: ./base_model.zip"
    echo "  Size: $(du -h base_model.zip | cut -f1)"
    echo ""
    echo "📤 Next steps:"
    echo "  1. Upload base_model.zip to Google Drive"
    echo "  2. Location: Projects/semantic-router/tools/config-gen/training/"
    echo "  3. Open colab_training_notebook.ipynb in Google Colab"
    echo ""
else
    echo "❌ Error: base_model directory not found"
    exit 1
fi

