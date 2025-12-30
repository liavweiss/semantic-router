#!/usr/bin/env python3
"""
Augment existing configurations by modifying them.

This script creates variations by adding/removing features, changing parameters,
and combining sections from different configs.
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import copy
import random
from tqdm import tqdm


AUGMENTATION_TYPES = [
    "add_semantic_cache",
    "remove_semantic_cache",
    "add_pii_detection",
    "remove_pii_detection",
    "add_tools",
    "remove_tools",
    "vary_thresholds",
    "add_decision",
    "remove_decision",
    "change_model",
    "change_endpoint",
    "add_observability",
    "simplify_config",
]


def validate_yaml_syntax(config_text: str) -> bool:
    """Validate YAML syntax."""
    try:
        yaml.safe_load(config_text)
        return True
    except yaml.YAMLError:
        return False


def augment_config(
    config_dict: Dict[str, Any], augmentation_type: str
) -> Dict[str, Any]:
    """
    Create variation of existing config.

    Args:
        config_dict: Original config as dict
        augmentation_type: Type of augmentation to apply

    Returns:
        Augmented config dict
    """
    augmented = copy.deepcopy(config_dict)

    if augmentation_type == "add_semantic_cache":
        if "semantic_cache" not in augmented or not augmented.get(
            "semantic_cache", {}
        ).get("enabled"):
            augmented["semantic_cache"] = {
                "enabled": True,
                "backend_type": random.choice(["memory", "milvus", "redis"]),
                "similarity_threshold": round(random.uniform(0.7, 0.9), 2),
                "max_entries": 1000,
                "ttl_seconds": 3600,
            }

    elif augmentation_type == "remove_semantic_cache":
        if "semantic_cache" in augmented:
            augmented["semantic_cache"]["enabled"] = False

    elif augmentation_type == "add_pii_detection":
        if "prompt_guard" not in augmented or not augmented.get("prompt_guard", {}).get(
            "enabled"
        ):
            augmented["prompt_guard"] = {
                "enabled": True,
                "threshold": round(random.uniform(0.6, 0.8), 2),
                "use_modernbert": True,
            }

    elif augmentation_type == "remove_pii_detection":
        if "prompt_guard" in augmented:
            augmented["prompt_guard"]["enabled"] = False

    elif augmentation_type == "add_tools":
        if "tools" not in augmented or not augmented.get("tools", {}).get("enabled"):
            augmented["tools"] = {
                "enabled": True,
                "top_k": 3,
                "similarity_threshold": 0.2,
                "tools_db_path": "config/tools_db.json",
                "fallback_to_empty": True,
            }

    elif augmentation_type == "remove_tools":
        if "tools" in augmented:
            augmented["tools"]["enabled"] = False

    elif augmentation_type == "vary_thresholds":
        if "semantic_cache" in augmented and augmented.get("semantic_cache", {}).get(
            "enabled"
        ):
            augmented["semantic_cache"]["similarity_threshold"] = round(
                random.uniform(0.7, 0.9), 2
            )
        if "prompt_guard" in augmented and augmented.get("prompt_guard", {}).get(
            "enabled"
        ):
            augmented["prompt_guard"]["threshold"] = round(random.uniform(0.6, 0.8), 2)

    elif augmentation_type == "add_decision":
        if "decisions" not in augmented:
            augmented["decisions"] = []

        # Get existing model names
        model_names = list(augmented.get("model_config", {}).keys())
        if not model_names:
            model_names = ["qwen3"]  # Default

        new_decision = {
            "name": f"decision_{len(augmented['decisions']) + 1}",
            "modelRefs": [
                {
                    "model": random.choice(model_names),
                    "use_reasoning": True,
                }
            ],
            "signals": [
                {
                    "type": random.choice(["domain", "keyword", "embedding"]),
                    "categories": random.sample(
                        ["business", "technical", "general", "support"], k=2
                    ),
                }
            ],
        }
        augmented["decisions"].append(new_decision)

    elif augmentation_type == "remove_decision":
        if "decisions" in augmented and len(augmented["decisions"]) > 1:
            augmented["decisions"].pop()

    elif augmentation_type == "change_model":
        # Change model name in decisions and model_config
        model_names = ["qwen3", "llama3", "phi4", "mistral"]
        if "model_config" in augmented and augmented["model_config"]:
            old_model = list(augmented["model_config"].keys())[0]
            new_model = random.choice([m for m in model_names if m != old_model])

            # Update model_config
            augmented["model_config"][new_model] = augmented["model_config"].pop(
                old_model
            )
            augmented["model_config"][new_model]["reasoning_family"] = new_model

            # Update decisions
            if "decisions" in augmented:
                for decision in augmented["decisions"]:
                    for model_ref in decision.get("modelRefs", []):
                        if model_ref.get("model") == old_model:
                            model_ref["model"] = new_model

    elif augmentation_type == "change_endpoint":
        if "vllm_endpoints" in augmented and augmented["vllm_endpoints"]:
            endpoint = augmented["vllm_endpoints"][0]
            endpoint["address"] = random.choice(
                ["172.28.0.20", "192.168.1.100", "10.0.0.50"]
            )
            endpoint["port"] = random.choice([8002, 11434, 8080])

    elif augmentation_type == "add_observability":
        # Add observability config if not present
        if "observability" not in augmented:
            augmented["observability"] = {
                "tracing": {
                    "enabled": True,
                    "jaeger_endpoint": "http://jaeger:14268/api/traces",
                },
                "metrics": {
                    "enabled": True,
                    "prometheus_endpoint": "http://prometheus:9090",
                },
            }

    elif augmentation_type == "simplify_config":
        # Keep only essential sections
        essential_keys = ["decisions", "model_config", "vllm_endpoints"]
        simplified = {k: v for k, v in augmented.items() if k in essential_keys}
        # Add minimal required config
        simplified["semantic_cache"] = {"enabled": False}
        simplified["prompt_guard"] = {"enabled": False}
        augmented = simplified

    return augmented


def generate_intent_from_augmentation(base_intent: str, augmentation_type: str) -> str:
    """Generate intent description from augmentation."""
    augmentation_descriptions = {
        "add_semantic_cache": "with semantic caching added",
        "remove_semantic_cache": "with semantic caching removed",
        "add_pii_detection": "with PII detection added",
        "remove_pii_detection": "with PII detection removed",
        "add_tools": "with tools auto-selection added",
        "remove_tools": "with tools removed",
        "vary_thresholds": "with modified thresholds",
        "add_decision": "with additional routing decision",
        "remove_decision": "with simplified routing",
        "change_model": "with different model",
        "change_endpoint": "with modified endpoints",
        "add_observability": "with observability added",
        "simplify_config": "simplified configuration",
    }

    desc = augmentation_descriptions.get(augmentation_type, "modified")
    return f"{base_intent} {desc}"


def main():
    parser = argparse.ArgumentParser(description="Augment existing config examples")
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSONL file with real examples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="raw/augmented_examples.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=100,
        help="Number of augmentations to generate",
    )
    parser.add_argument(
        "--augmentation-types",
        type=str,
        nargs="+",
        default=AUGMENTATION_TYPES,
        help="Types of augmentations to apply",
    )

    args = parser.parse_args()

    # Load input examples
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    base_examples = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                base_examples.append(json.loads(line))

    print(f"Loaded {len(base_examples)} base examples")

    # Generate augmentations
    augmented_examples = []

    with tqdm(total=args.num_augmentations) as pbar:
        for _ in range(args.num_augmentations):
            # Pick random base example
            base = random.choice(base_examples)

            # Pick random augmentation type
            aug_type = random.choice(args.augmentation_types)

            # Parse base config
            try:
                base_config_dict = yaml.safe_load(
                    base.get("full_config", base.get("config", ""))
                )
                if not base_config_dict:
                    pbar.update(1)
                    continue
            except yaml.YAMLError:
                pbar.update(1)
                continue

            # Apply augmentation
            augmented_dict = augment_config(base_config_dict, aug_type)

            # Convert back to YAML
            try:
                augmented_yaml = yaml.dump(
                    augmented_dict, default_flow_style=False, sort_keys=False
                )
            except Exception as e:
                print(f"Warning: Could not serialize augmented config: {e}")
                pbar.update(1)
                continue

            # Validate
            if not validate_yaml_syntax(augmented_yaml):
                pbar.update(1)
                continue

            # Create augmented example
            augmented_example = {
                "id": f"augmented_{base.get('id', 'unknown')}_{hash(augmented_yaml) % 10000}",
                "config": augmented_yaml,
                "intent": generate_intent_from_augmentation(
                    base.get("intent", "Configuration"), aug_type
                ),
                "deployment_context": base.get("deployment_context", "unknown"),
                "source": "augmented",
                "base_id": base.get("id", "unknown"),
                "augmentation_type": aug_type,
            }

            augmented_examples.append(augmented_example)
            pbar.update(1)

    print(f"\nGenerated {len(augmented_examples)} augmented examples")

    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in augmented_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Written to {output_path}")

    # Print statistics
    print("\nStatistics:")
    aug_types = {}
    for ex in augmented_examples:
        aug_type = ex.get("augmentation_type", "unknown")
        aug_types[aug_type] = aug_types.get(aug_type, 0) + 1
    print(f"  By augmentation type: {aug_types}")

    return 0


if __name__ == "__main__":
    exit(main())
