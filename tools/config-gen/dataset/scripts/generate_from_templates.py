#!/usr/bin/env python3
"""
Generate configuration variations from templates.

This script creates systematic variations by filling template placeholders with
different values.
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from itertools import product
import random
from tqdm import tqdm


# Variation parameters
VARIATIONS = {
    "decision_name": [
        "business_decision",
        "technical_decision",
        "general_decision",
        "support_decision",
    ],
    "model_name": ["qwen3", "llama3", "phi4", "mistral"],
    "signal_type": ["domain", "keyword", "embedding"],
    "category1": ["business", "technical", "general", "support"],
    "category2": ["finance", "engineering", "customer_service", "sales"],
    "endpoint_address": ["172.28.0.20", "192.168.1.100", "10.0.0.50", "172.16.0.10"],
    "endpoint_port": [8002, 11434, 8080, 8000],
    "endpoint_weight": [1, 2, 3],
    "cache_enabled": [True, False],
    "cache_backend": ["memory", "milvus", "redis"],
    "cache_threshold": [0.7, 0.8, 0.9],
    "guard_enabled": [True, False],
    "guard_threshold": [0.6, 0.7, 0.8],
    "tools_enabled": [True, False],
    "deployment_context": ["quickstart", "ai-gateway", "istio", "kubernetes"],
}


def create_base_template() -> str:
    """Create base config template."""
    return """# Semantic Router Configuration
# Generated from template

decisions:
  - name: "{decision_name}"
    modelRefs:
      - model: "{model_name}"
        use_reasoning: true
    signals:
      - type: "{signal_type}"
        categories:
          - "{category1}"
          - "{category2}"

model_config:
  {model_name}:
    reasoning_family: "{model_name}"
    preferred_endpoints:
      - "{endpoint_name}"

vllm_endpoints:
  - name: "{endpoint_name}"
    address: "{endpoint_address}"
    port: {endpoint_port}
    weight: {endpoint_weight}

semantic_cache:
  enabled: {cache_enabled}
  backend_type: "{cache_backend}"
  similarity_threshold: {cache_threshold}
  max_entries: 1000
  ttl_seconds: 3600

prompt_guard:
  enabled: {guard_enabled}
  threshold: {guard_threshold}
  use_modernbert: true

tools:
  enabled: {tools_enabled}
  top_k: 3
  similarity_threshold: 0.2
"""


def fill_template(template: str, values: Dict[str, Any]) -> str:
    """Fill template with values."""
    config = template
    for key, value in values.items():
        placeholder = "{" + key + "}"
        if placeholder in config:
            config = config.replace(placeholder, str(value))

    # Set endpoint_name based on model_name
    if "{endpoint_name}" in config:
        endpoint_name = f"endpoint_{values.get('model_name', 'default')}"
        config = config.replace("{endpoint_name}", endpoint_name)

    return config


def validate_yaml_syntax(config_text: str) -> bool:
    """Validate YAML syntax."""
    try:
        yaml.safe_load(config_text)
        return True
    except yaml.YAMLError:
        return False


def generate_intent_from_variation(values: Dict[str, Any]) -> str:
    """Generate intent description from variation values."""
    intent_parts = []

    # Model
    intent_parts.append(f"Route queries to {values.get('model_name', 'model')}")

    # Signal type
    signal_type = values.get("signal_type", "domain")
    if signal_type == "domain":
        intent_parts.append("using domain-based routing")
    elif signal_type == "keyword":
        intent_parts.append("using keyword-based routing")
    else:
        intent_parts.append("using embedding-based routing")

    # Cache
    if values.get("cache_enabled"):
        intent_parts.append(
            f"with {values.get('cache_backend', 'memory')} semantic cache"
        )

    # Guard
    if values.get("guard_enabled"):
        intent_parts.append("with PII detection")

    # Tools
    if values.get("tools_enabled"):
        intent_parts.append("with tools auto-selection")

    # Deployment
    deployment = values.get("deployment_context", "quickstart")
    intent_parts.append(f"for {deployment} deployment")

    return ", ".join(intent_parts)


def generate_template_variations(
    template: str, max_variations: int = 200, sample_strategy: str = "random"
) -> List[Dict[str, Any]]:
    """
    Generate config variations from template.

    Args:
        template: Template string with placeholders
        max_variations: Maximum number of variations to generate
        sample_strategy: "random" or "systematic"
    """
    variations = []

    # Get all parameter names
    param_names = list(VARIATIONS.keys())
    param_values = [VARIATIONS[name] for name in param_names]

    if sample_strategy == "random":
        # Random sampling
        for _ in range(max_variations):
            values = {}
            for name in param_names:
                values[name] = random.choice(VARIATIONS[name])

            # Fill template
            config = fill_template(template, values)

            # Validate
            if validate_yaml_syntax(config):
                variations.append(
                    {
                        "id": f"template_{hash(config) % 100000}",
                        "config": config,
                        "intent": generate_intent_from_variation(values),
                        "deployment_context": values.get(
                            "deployment_context", "quickstart"
                        ),
                        "source": "template",
                        "values": values,
                    }
                )
    else:
        # Systematic (sample from product)
        total_combinations = 1
        for vals in param_values:
            total_combinations *= len(vals)

        # Sample if too many
        if total_combinations > max_variations:
            # Use random sampling
            seen = set()
            for _ in range(max_variations):
                values = {}
                for name in param_names:
                    values[name] = random.choice(VARIATIONS[name])

                # Check for duplicates
                values_tuple = tuple(sorted(values.items()))
                if values_tuple in seen:
                    continue
                seen.add(values_tuple)

                config = fill_template(template, values)
                if validate_yaml_syntax(config):
                    variations.append(
                        {
                            "id": f"template_{hash(config) % 100000}",
                            "config": config,
                            "intent": generate_intent_from_variation(values),
                            "deployment_context": values.get(
                                "deployment_context", "quickstart"
                            ),
                            "source": "template",
                            "values": values,
                        }
                    )
        else:
            # Generate all combinations
            for combo in product(*param_values):
                values = dict(zip(param_names, combo))
                config = fill_template(template, values)
                if validate_yaml_syntax(config):
                    variations.append(
                        {
                            "id": f"template_{hash(config) % 100000}",
                            "config": config,
                            "intent": generate_intent_from_variation(values),
                            "deployment_context": values.get(
                                "deployment_context", "quickstart"
                            ),
                            "source": "template",
                            "values": values,
                        }
                    )

    return variations


def main():
    parser = argparse.ArgumentParser(
        description="Generate config variations from templates"
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to template file (default: use built-in template)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="raw/template_examples.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-variations",
        type=int,
        default=200,
        help="Maximum number of variations to generate",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        choices=["random", "systematic"],
        default="random",
        help="Sampling strategy",
    )

    args = parser.parse_args()

    # Load or create template
    if args.template:
        template_path = Path(args.template)
        if not template_path.exists():
            print(f"Error: Template file not found: {template_path}")
            return 1
        with open(template_path, "r") as f:
            template = f.read()
    else:
        template = create_base_template()
        print("Using built-in template")

    print(f"Generating {args.max_variations} template variations...")

    # Generate variations
    variations = generate_template_variations(
        template,
        max_variations=args.max_variations,
        sample_strategy=args.sample_strategy,
    )

    print(f"Generated {len(variations)} valid variations")

    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for variation in variations:
            f.write(json.dumps(variation) + "\n")

    print(f"Written to {output_path}")

    # Print statistics
    print("\nStatistics:")
    contexts = {}
    for var in variations:
        ctx = var.get("deployment_context", "unknown")
        contexts[ctx] = contexts.get(ctx, 0) + 1
    print(f"  By deployment context: {contexts}")

    return 0


if __name__ == "__main__":
    exit(main())
