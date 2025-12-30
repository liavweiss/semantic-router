#!/usr/bin/env python3
"""
Collect real configuration files from the semantic-router codebase.

This script scans the codebase for all YAML configuration files and collects them
with metadata for dataset creation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import argparse


# Directories to search for config files
CONFIG_DIRS = [
    "config",
    "e2e/profiles",
    "deploy/kubernetes",
    "bench",
]

# Config file patterns to look for
CONFIG_PATTERNS = [
    "config.yaml",
    "*.yaml",
    "*.yml",
]


def find_config_files(base_dir: str) -> List[Dict[str, str]]:
    """
    Find all configuration files in the codebase.

    Args:
        base_dir: Base directory of semantic-router project

    Returns:
        List of dicts with file info: {path, relative_path, category}
    """
    config_files = []
    base_path = Path(base_dir)

    for config_dir in CONFIG_DIRS:
        dir_path = base_path / config_dir

        if not dir_path.exists():
            continue

        # Determine category
        if "config" in config_dir:
            category = "main"
        elif "e2e" in config_dir:
            category = "e2e"
        elif "deploy" in config_dir:
            category = "deploy"
        elif "bench" in config_dir:
            category = "bench"
        else:
            category = "other"

        # Find all YAML files
        for yaml_file in dir_path.rglob("*.yaml"):
            # Skip certain files
            if any(
                skip in str(yaml_file)
                for skip in ["kustomization", "values", "deployment", "service"]
            ):
                continue

            # Check if it's a semantic-router config (has key fields)
            try:
                with open(yaml_file, "r") as f:
                    content = f.read()
                    # Quick check if it looks like a router config
                    if any(
                        keyword in content.lower()
                        for keyword in [
                            "decisions",
                            "model_config",
                            "vllm_endpoints",
                            "semantic_cache",
                            "prompt_guard",
                            "router",
                        ]
                    ):
                        config_files.append(
                            {
                                "path": str(yaml_file.absolute()),
                                "relative_path": str(yaml_file.relative_to(base_path)),
                                "category": category,
                                "name": yaml_file.stem,
                            }
                        )
            except Exception as e:
                print(f"Warning: Could not read {yaml_file}: {e}")
                continue

    return config_files


def extract_deployment_context(relative_path: str) -> str:
    """Extract deployment context from file path."""
    if "quickstart" in relative_path or "config/config.yaml" in relative_path:
        return "quickstart"
    elif "ai-gateway" in relative_path:
        return "ai-gateway"
    elif "istio" in relative_path:
        return "istio"
    elif "aibrix" in relative_path:
        return "aibrix"
    elif "llm-d" in relative_path or "llmd" in relative_path:
        return "llm-d"
    elif "production" in relative_path:
        return "production-stack"
    elif "dynamic-config" in relative_path:
        return "dynamic-config"
    elif "routing-strategies" in relative_path:
        return "routing-strategies"
    else:
        return "unknown"


def extract_features(config_content: str) -> List[str]:
    """Extract key features from config content."""
    features = []
    content_lower = config_content.lower()

    if "semantic_cache" in content_lower and "enabled: true" in content_lower:
        features.append("semantic_cache")
    if "prompt_guard" in content_lower and "enabled: true" in content_lower:
        features.append("pii_detection")
    if "jailbreak" in content_lower:
        features.append("jailbreak_detection")
    if "decisions:" in content_lower:
        features.append("routing_decisions")
    if "tools:" in content_lower and "enabled: true" in content_lower:
        features.append("tools")
    if "observability" in content_lower or "tracing" in content_lower:
        features.append("observability")
    if "lora" in content_lower:
        features.append("lora_routing")

    return features


def collect_config(config_file: Dict[str, str]) -> Dict[str, Any]:
    """
    Collect and annotate a single config file.

    Args:
        config_file: File info dict

    Returns:
        Annotated config dict
    """
    try:
        with open(config_file["path"], "r") as f:
            config_content = f.read()

        # Parse YAML to validate
        try:
            config_dict = yaml.safe_load(config_content)
        except yaml.YAMLError as e:
            print(f"Warning: Invalid YAML in {config_file['path']}: {e}")
            return None

        # Extract metadata
        deployment_context = extract_deployment_context(config_file["relative_path"])
        features = extract_features(config_content)

        # Determine complexity
        complexity = "low"
        if len(features) > 3:
            complexity = "high"
        elif len(features) > 1:
            complexity = "medium"

        # Generate basic intent (will be improved manually later)
        intent_parts = []
        if deployment_context != "unknown":
            intent_parts.append(f"{deployment_context} deployment")
        if "semantic_cache" in features:
            intent_parts.append("with semantic caching")
        if "routing_decisions" in features:
            intent_parts.append("with routing decisions")
        if "pii_detection" in features:
            intent_parts.append("with PII detection")

        intent = (
            "Configuration " + ", ".join(intent_parts)
            if intent_parts
            else "Basic configuration"
        )

        return {
            "id": f"real_{config_file['name']}_{hash(config_file['relative_path']) % 10000}",
            "source_file": config_file["relative_path"],
            "category": config_file["category"],
            "deployment_context": deployment_context,
            "intent": intent,
            "use_case": f"Configuration for {deployment_context} deployment",
            "complexity": complexity,
            "key_features": features,
            "full_config": config_content,
            "source": "real",
            "collected_at": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error processing {config_file['path']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect real config files from codebase"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory of semantic-router project (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="raw/real_examples.jsonl",
        help="Output JSONL file path",
    )

    args = parser.parse_args()

    # Resolve base directory
    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return 1

    print(f"Scanning for config files in: {base_dir}")

    # Find config files
    config_files = find_config_files(str(base_dir))
    print(f"Found {len(config_files)} config files")

    # Collect and annotate
    collected_configs = []
    for config_file in config_files:
        annotated = collect_config(config_file)
        if annotated:
            collected_configs.append(annotated)

    print(f"Successfully collected {len(collected_configs)} configs")

    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for config in collected_configs:
            f.write(json.dumps(config) + "\n")

    print(f"Written {len(collected_configs)} configs to {output_path}")

    # Print statistics
    print("\nStatistics:")
    print(f"  Total configs: {len(collected_configs)}")
    categories = {}
    contexts = {}
    for config in collected_configs:
        categories[config["category"]] = categories.get(config["category"], 0) + 1
        contexts[config["deployment_context"]] = (
            contexts.get(config["deployment_context"], 0) + 1
        )

    print(f"  By category: {categories}")
    print(f"  By deployment context: {contexts}")

    return 0


if __name__ == "__main__":
    exit(main())
