#!/usr/bin/env python3
"""
Semantic Router Configuration Generation Fine-tuning with LoRA

Fine-tunes a base LLM (Llama 3.1 8B or Qwen 2.5 7B) to generate semantic-router
YAML configurations from natural language descriptions using LoRA.

Usage:
    # Train with Llama 3.1 8B
    python train_config_gen_lora.py --mode train --base-model meta-llama/Llama-3.1-8B-Instruct --epochs 3

    # Train with Qwen 2.5 7B
    python train_config_gen_lora.py --mode train --base-model Qwen/Qwen2.5-7B-Instruct --epochs 3

    # Train with local model
    python train_config_gen_lora.py --mode train --base-model ./llama3_8b_base --epochs 3

    # Test inference
    python train_config_gen_lora.py --mode test --model-path config_gen_llama3_8b_v1.0.0

Features:
    - LoRA (Low-Rank Adaptation) for efficient fine-tuning
    - Instruction-following format (instruction/input/output)
    - Automatic dataset loading from JSONL
    - Validation during training
    - Model versioning
    - GPU/CPU automatic detection
"""

import json
import logging
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSONL file."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
    """
    Format training prompt with few-shot examples.

    Shows the model 2 examples of correct YAML generation before the actual task.
    This teaches the model to output pure YAML without explanations or markdown.
    """
    # Few-shot examples to teach the model the correct format
    few_shot_examples = """You are an expert in generating semantic-router YAML configurations.

IMPORTANT RULES:
- Output ONLY valid YAML configuration
- Do NOT add explanations, comments, or markdown fences
- Do NOT add "Here's the config" or similar text
- Start directly with the YAML configuration

Example 1:
Request: Create a basic decision for routing business queries to qwen3 model
Output:
decisions:
  - name: business_decision
    description: Route business queries
    priority: 100
    rules:
      operator: AND
      conditions:
        - type: domain
          name: business
    modelRefs:
      - model: vllm-qwen3-8b

Example 2:
Request: Enable semantic cache with memory backend
Output:
semantic_cache:
  enabled: true
  backend_type: memory
  similarity_threshold: 0.85
  max_entries: 1000
  ttl_seconds: 3600

"""

    # Now add the actual task
    if input_text:
        prompt = f"{few_shot_examples}\nNow generate configuration for:\nRequest: {instruction}\nContext: {input_text}\nOutput:\n{output}"
    else:
        prompt = f"{few_shot_examples}\nNow generate configuration for:\nRequest: {instruction}\nOutput:\n{output}"

    return prompt


def preprocess_function(examples, tokenizer, max_length: int = 4096):
    """Preprocess examples for training."""
    # Format prompts
    prompts = []

    # Determine instruction field
    if "instruction" in examples:
        instruction_field = "instruction"
    elif "intent" in examples:
        instruction_field = "intent"
    else:
        raise ValueError(
            f"No instruction field found. Available fields: {list(examples.keys())}"
        )

    # Determine output field (config content)
    if "output" in examples:
        output_field = "output"
    elif "full_config" in examples:
        output_field = "full_config"
    elif "config" in examples:
        output_field = "config"
    else:
        raise ValueError(
            f"No output field found. Available fields: {list(examples.keys())}"
        )

    for i in range(len(examples[instruction_field])):
        prompt = format_prompt(
            examples[instruction_field][i],
            examples.get("input", [""] * len(examples[instruction_field]))[i],
            examples[output_field][i],
        )
        prompts.append(prompt)

    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # Note: We DON'T set labels here!
    # DataCollatorForLanguageModeling will automatically create labels from input_ids
    # for causal LM (mlm=False) during batch collation

    return tokenized


def create_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """Create LoRA configuration."""
    # Default target modules for Llama/Qwen
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )


def clean_generated_output(text: str) -> str:
    """
    Post-process model output to extract pure YAML.

    Strips:
    - Markdown code fences (```yaml, ```)
    - Explanatory text before/after YAML
    - Common phrases like "Here's", "Here is"
    """
    # Remove markdown code fences
    text = text.replace("```yaml", "").replace("```", "")

    # Remove common prefixes
    prefixes_to_remove = [
        "Here's the configuration:",
        "Here is the configuration:",
        "Here's the YAML:",
        "Here is the YAML:",
        "Configuration:",
        "YAML Configuration:",
        "Output:",
    ]

    for prefix in prefixes_to_remove:
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix) :].strip()

    # Find the first line that looks like YAML (starts with a key)
    lines = text.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # YAML typically starts with a key (word followed by colon) or list dash
        if stripped and (
            (":" in stripped and not stripped.startswith("#"))
            or stripped.startswith("-")
        ):
            start_idx = i
            break

    # Take from first YAML line to end, but stop at explanatory text
    yaml_lines = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        # Stop if we hit explanatory text
        if any(
            phrase in stripped.lower()
            for phrase in ["explanation:", "note:", "this configuration", "you can"]
        ):
            break
        yaml_lines.append(line)

    result = "\n".join(yaml_lines).strip()
    return result


def validate_generated_config(config_text: str) -> tuple[bool, str]:
    """Validate generated YAML config."""
    try:
        yaml.safe_load(config_text)
        return True, ""
    except yaml.YAMLError as e:
        return False, str(e)


def evaluate_model(
    test_file: str,
    base_model: Optional[str] = None,
    model_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    model_name: str = "Model",
):
    """
    Evaluate model on test set and calculate metrics.

    Args:
        test_file: Path to test JSONL file
        base_model: Base model path (for baseline evaluation or when model_path is LoRA adapter)
        model_path: Trained model path (LoRA adapter or full model). If None, evaluates base_model only.
        max_samples: Maximum number of samples to evaluate
        model_name: Name to display in output (e.g., "Base Model" or "Trained Model")
    """
    # Determine which model to load
    if model_path:
        logger.info(f"Evaluating trained model from {model_path} on {test_file}")
        # Load base model if LoRA adapter
        if base_model:
            logger.info(f"Loading base model: {base_model}")
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)

            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model_obj, model_path)
            model = model.merge_and_unload()
        else:
            # Assume full model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif base_model:
        logger.info(f"Evaluating base model {base_model} on {test_file} (baseline)")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        raise ValueError("Either base_model or model_path must be provided")

    model.eval()

    # Load test examples
    test_examples = load_jsonl_dataset(test_file)
    if max_samples:
        test_examples = test_examples[:max_samples]

    logger.info(f"Evaluating on {len(test_examples)} test examples...")

    # Metrics
    total = len(test_examples)
    valid_yaml_count = 0
    errors = []

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} EVALUATION - METRICS")
    print("=" * 70)

    for i, example in enumerate(test_examples):
        # Support multiple field name variations
        instruction = example.get("instruction") or example.get("intent", "")
        expected_output = (
            example.get("output")
            or example.get("full_config")
            or example.get("config", "")
        )

        prompt = format_prompt(instruction, "", "")
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3072,  # Increased for complete configs
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,  # Prevent token repetition loops
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract YAML
        if "Output:" in generated_text:
            config_text = generated_text.split("Output:")[-1].strip()
        else:
            config_text = generated_text[len(prompt) :].strip()

        # NOTE: We do NOT use post-processing during evaluation
        # to get the TRUE model accuracy. Post-processing should only
        # be used in production API as a safety net.

        # Validate YAML syntax (raw model output)
        is_valid, error = validate_generated_config(config_text)
        if is_valid:
            valid_yaml_count += 1
        else:
            errors.append(
                {"example": i, "instruction": instruction[:50], "error": error}
            )

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{total} examples...")

    # Calculate metrics
    yaml_accuracy = (valid_yaml_count / total) * 100
    error_rate = ((total - valid_yaml_count) / total) * 100

    results = {
        "model_name": model_name,
        "total": total,
        "valid_yaml": valid_yaml_count,
        "invalid_yaml": total - valid_yaml_count,
        "yaml_accuracy": yaml_accuracy,
        "error_rate": error_rate,
        "success_rate": yaml_accuracy,
        "sample_errors": errors[:5] if errors else [],
    }

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} - EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total examples: {total}")
    print(f"Valid YAML: {valid_yaml_count}")
    print(f"Invalid YAML: {total - valid_yaml_count}")
    print(f"\n📊 Metrics:")
    print(f"  YAML Syntax Accuracy: {yaml_accuracy:.2f}%")
    print(f"  Error Rate: {error_rate:.2f}%")
    print(f"  Success Rate: {yaml_accuracy:.2f}%")

    if errors:
        print(f"\n❌ Sample Errors (first 5):")
        for err in errors[:5]:
            print(f"  Example {err['example']}: {err['instruction']}...")
            print(f"    Error: {err['error'][:100]}...")

    # Save results to file for comparison
    # Clean model name for filename (remove special chars, spaces, parentheses)
    clean_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )
    results_file = f"evaluation_results_{clean_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

    return results


def test_model(
    model_path: str,
    base_model: Optional[str] = None,
    test_queries: Optional[List[tuple[str, str]]] = None,
):
    """Test trained model with sample queries."""
    logger.info(f"Loading model from {model_path}")

    # Load base model if LoRA adapter
    if base_model:
        logger.info(f"Loading base model: {base_model}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model_obj, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    # Default test queries
    if test_queries is None:
        test_queries = [
            ("Route business queries to qwen3 model", "business_decision"),
            ("Enable semantic caching with memory backend", "semantic_cache"),
            ("Create configuration for quickstart deployment", "full_config"),
        ]

    logger.info("Testing model with sample queries...")
    print("\n" + "=" * 70)
    print("CONFIG GENERATION MODEL - TEST RESULTS")
    print("=" * 70)

    valid_count = 0
    for query, expected_section in test_queries:
        prompt = format_prompt(query, "", "")
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,  # Prevent token repetition loops
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract YAML from output (after "Output:")
        if "Output:" in generated_text:
            config_text = generated_text.split("Output:")[-1].strip()
        else:
            config_text = generated_text[len(prompt) :].strip()

        # NOTE: We do NOT use post-processing during testing
        # to see the TRUE model output quality.

        # Validate
        is_valid, error = validate_generated_config(config_text)
        if is_valid:
            valid_count += 1
        status = "✅" if is_valid else "❌"

        print(f"\n{status} Query: {query}")
        print(f"   Expected section: {expected_section}")
        print(f"   Valid YAML: {is_valid}")
        if not is_valid:
            print(f"   Error: {error}")
        print(f"   Generated (first 200 chars): {config_text[:200]}...")

    print(
        f"\n📊 Quick Test Results: {valid_count}/{len(test_queries)} valid ({valid_count/len(test_queries)*100:.1f}%)"
    )


def train_model(
    train_file: str,
    val_file: str,
    base_model: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_length: int = 4096,
    gradient_accumulation_steps: int = 4,
):
    """Train model with LoRA."""
    logger.info(f"Loading dataset from {train_file}")

    # Load datasets
    train_examples = load_jsonl_dataset(train_file)
    val_examples = load_jsonl_dataset(val_file)

    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    # Load tokenizer and model
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Apply LoRA
    logger.info("Applying LoRA configuration...")
    lora_config = create_lora_config(rank=lora_rank, alpha=lora_alpha)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",  # Disable intermediate checkpoints
        save_only_model=True,  # Only save model weights, not optimizer state
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model (bypass model card creation due to permissions)
    logger.info(f"Saving model to {output_dir}")
    os.makedirs(output_dir, mode=0o755, exist_ok=True)

    # Save PEFT model directly (skip model card)
    model.save_pretrained(
        output_dir, safe_serialization=True, create_pr=False, push_to_hub=False
    )
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "version": "v1.0.0",
        "base_model": base_model,
        "training_date": datetime.now().isoformat(),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train config generation model with LoRA"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "evaluate"],
        required=True,
        help="Mode: train, test, or evaluate",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="dataset/processed/train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="dataset/processed/val.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to trained model (for test mode)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="dataset/processed/test.jsonl",
        help="Path to test JSONL file (for evaluate mode)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Model",
        help="Name to display in evaluation output (e.g., 'Base Model' or 'Trained Model')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/config_gen_llama3_8b_v1.0.0",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            train_file=args.train_file,
            val_file=args.val_file,
            base_model=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            max_length=args.max_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    elif args.mode == "test":
        if not args.model_path:
            args.model_path = args.output_dir
        test_model(
            model_path=args.model_path,
            base_model=args.base_model,
        )
    elif args.mode == "evaluate":
        # If model_path not provided, evaluate base model only (baseline)
        if not args.model_path:
            if not args.base_model:
                # Try to use output_dir as model_path
                args.model_path = args.output_dir
            else:
                # Evaluate base model as baseline
                evaluate_model(
                    test_file=args.test_file,
                    base_model=args.base_model,
                    max_samples=args.max_samples,
                    model_name=args.model_name,
                )
                return

        evaluate_model(
            test_file=args.test_file,
            base_model=args.base_model,
            model_path=args.model_path,
            max_samples=args.max_samples,
            model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
