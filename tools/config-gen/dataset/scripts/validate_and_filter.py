#!/usr/bin/env python3
"""
Validate and filter generated configuration examples.

This script validates YAML syntax and optionally runs semantic-router validator
to filter out invalid configurations.
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import subprocess
import tempfile
import os


def validate_yaml_syntax(config_text: str) -> Tuple[bool, str]:
    """
    Validate YAML syntax.

    Returns:
        (is_valid, error_message)
    """
    try:
        yaml.safe_load(config_text)
        return True, ""
    except yaml.YAMLError as e:
        return False, str(e)


def validate_with_router_validator(
    config_text: str, validator_path: str = None
) -> Tuple[bool, str]:
    """
    Validate config using semantic-router validator (if available).

    Args:
        config_text: Config YAML text
        validator_path: Path to validator binary (optional)

    Returns:
        (is_valid, error_message)
    """
    # For now, we'll do basic validation
    # TODO: Integrate with actual semantic-router validator if available

    try:
        config_dict = yaml.safe_load(config_text)
        if not config_dict:
            return False, "Config is empty"

        # Basic checks
        # Check if decisions reference models that exist
        if "decisions" in config_dict:
            decisions = config_dict.get("decisions", [])
            model_config = config_dict.get("model_config", {})

            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                model_refs = decision.get("modelRefs", [])
                for model_ref in model_refs:
                    if isinstance(model_ref, dict):
                        model_name = model_ref.get("model")
                        if model_name and model_name not in model_config:
                            return (
                                False,
                                f"Decision references model '{model_name}' not in model_config",
                            )

        # Check if vllm_endpoints exist if model_config references them
        if "model_config" in config_dict:
            model_config = config_dict["model_config"]
            vllm_endpoints = config_dict.get("vllm_endpoints", [])
            endpoint_names = [
                ep.get("name") for ep in vllm_endpoints if isinstance(ep, dict)
            ]

            for model_name, model_cfg in model_config.items():
                if isinstance(model_cfg, dict):
                    preferred_endpoints = model_cfg.get("preferred_endpoints", [])
                    for ep_name in preferred_endpoints:
                        if ep_name not in endpoint_names:
                            return (
                                False,
                                f"Model '{model_name}' references endpoint '{ep_name}' not in vllm_endpoints",
                            )

        return True, ""
    except Exception as e:
        return False, str(e)


def validate_example(
    example: Dict[str, Any], validate_semantic: bool = True
) -> Tuple[bool, str]:
    """
    Validate a single example.

    Args:
        example: Example dict
        validate_semantic: Whether to run semantic validation

    Returns:
        (is_valid, error_message)
    """
    config_text = example.get("config") or example.get("full_config", "")
    if not config_text:
        return False, "No config content"

    # Validate YAML syntax
    is_valid, error = validate_yaml_syntax(config_text)
    if not is_valid:
        return False, f"YAML syntax error: {error}"

    # Validate semantic (if enabled)
    if validate_semantic:
        is_valid, error = validate_with_router_validator(config_text)
        if not is_valid:
            return False, f"Semantic validation error: {error}"

    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Validate and filter config examples")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input JSONL file(s) to validate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed/validated.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip semantic validation (only check YAML syntax)",
    )
    parser.add_argument(
        "--errors-output", type=str, help="Output file for invalid examples with errors"
    )

    args = parser.parse_args()

    # Load all examples
    all_examples = []
    for input_file in args.input:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}")
            continue

        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    all_examples.append(json.loads(line))

    print(f"Loaded {len(all_examples)} examples to validate")

    # Validate examples
    valid_examples = []
    invalid_examples = []

    for example in tqdm(all_examples, desc="Validating"):
        is_valid, error = validate_example(
            example, validate_semantic=not args.skip_semantic
        )

        if is_valid:
            valid_examples.append(example)
        else:
            invalid_example = example.copy()
            invalid_example["validation_error"] = error
            invalid_examples.append(invalid_example)

    print(f"\nValidation Results:")
    print(
        f"  Valid: {len(valid_examples)} ({len(valid_examples)/len(all_examples)*100:.1f}%)"
    )
    print(
        f"  Invalid: {len(invalid_examples)} ({len(invalid_examples)/len(all_examples)*100:.1f}%)"
    )

    # Write valid examples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in valid_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nWritten {len(valid_examples)} valid examples to {output_path}")

    # Write invalid examples (if requested)
    if args.errors_output and invalid_examples:
        errors_path = Path(args.errors_output)
        errors_path.parent.mkdir(parents=True, exist_ok=True)

        with open(errors_path, "w") as f:
            for example in invalid_examples:
                f.write(json.dumps(example) + "\n")

        print(f"Written {len(invalid_examples)} invalid examples to {errors_path}")

    # Print error statistics
    if invalid_examples:
        print("\nError Statistics:")
        error_types = {}
        for ex in invalid_examples:
            error = ex.get("validation_error", "unknown")
            error_type = error.split(":")[0] if ":" in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(
            error_types.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {error_type}: {count}")

    return 0


if __name__ == "__main__":
    exit(main())
