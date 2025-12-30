#!/usr/bin/env python3
"""
Quick evaluation script with robust generation parameters.
NO repetition loops, NO truncation, CLEAN output.
"""

import json
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def evaluate_model(base_model_path, lora_path, test_file, max_samples=50):
    """Evaluate model with robust parameters."""

    print(f"Loading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="auto"
    )

    print(f"Loading LoRA adapters from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    print(f"Loading test data from {test_file}...")
    test_data = []
    with open(test_file) as f:
        for line in f:
            test_data.append(json.loads(line))

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"\nEvaluating {len(test_data)} examples...")

    valid_count = 0
    results = []

    for idx, example in enumerate(test_data):
        instruction = example.get("instruction") or example.get("intent", "")

        # Simple prompt - NO few-shot, just direct instruction
        prompt = f"""Generate YAML configuration for this request. Output ONLY valid YAML, no explanations.

Request: {instruction}
YAML Config:
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,  # Shorter to avoid repetition
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.9,
                repetition_penalty=1.2,  # Strong penalty against repetition
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract YAML (everything after the prompt)
        config_text = generated_text[len(prompt) :].strip()

        # Clean up common issues
        if config_text.startswith("```yaml"):
            config_text = config_text[7:]
        if config_text.startswith("```"):
            config_text = config_text[3:]
        if config_text.endswith("```"):
            config_text = config_text[:-3]
        config_text = config_text.strip()

        # Validate YAML
        is_valid = False
        error_msg = None
        try:
            yaml.safe_load(config_text)
            is_valid = True
            valid_count += 1
        except Exception as e:
            error_msg = str(e)

        results.append(
            {
                "instruction": instruction,
                "generated": config_text[:500],  # Store first 500 chars
                "valid": is_valid,
                "error": error_msg,
            }
        )

        if (idx + 1) % 10 == 0:
            print(
                f"  Processed {idx + 1}/{len(test_data)} examples... (Valid so far: {valid_count})"
            )

    # Print results
    accuracy = (valid_count / len(test_data)) * 100
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total: {len(test_data)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {len(test_data) - valid_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"{'='*60}")

    # Show first valid example
    print(f"\n✅ FIRST VALID EXAMPLE:")
    for r in results:
        if r["valid"]:
            print(f"Instruction: {r['instruction']}")
            print(f"Generated:\n{r['generated']}")
            break
    else:
        print("No valid examples found!")

    # Show first invalid example
    print(f"\n❌ FIRST INVALID EXAMPLE:")
    for r in results:
        if not r["valid"]:
            print(f"Instruction: {r['instruction']}")
            print(f"Generated:\n{r['generated']}")
            print(f"Error: {r['error']}")
            break

    # Save results
    output_file = "quick_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "total": len(test_data),
                "valid": valid_count,
                "invalid": len(test_data) - valid_count,
                "accuracy": accuracy,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n💾 Results saved to: {output_file}")

    return accuracy


if __name__ == "__main__":
    accuracy = evaluate_model(
        base_model_path="./base_model",
        lora_path="./models/config_gen_v1.0.0",
        test_file="./test_50.jsonl",
        max_samples=50,
    )

    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY: {accuracy:.1f}%")
    print(f"{'='*60}")
