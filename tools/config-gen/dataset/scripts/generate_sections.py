#!/usr/bin/env python3
"""
Generate section-level examples from full configurations.

This script extracts individual sections from full configs to create examples
for partial config generation.
"""

import yaml
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def validate_yaml_syntax(config_text: str) -> bool:
    """Validate YAML syntax."""
    try:
        yaml.safe_load(config_text)
        return True
    except yaml.YAMLError:
        return False


def extract_section(config_dict: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    """Extract a specific section from config."""
    if section_name == "decisions":
        return {"decisions": config_dict.get("decisions", [])}
    elif section_name == "semantic_cache":
        return {"semantic_cache": config_dict.get("semantic_cache", {})}
    elif section_name == "model_config":
        return {"model_config": config_dict.get("model_config", {})}
    elif section_name == "vllm_endpoints":
        return {"vllm_endpoints": config_dict.get("vllm_endpoints", [])}
    elif section_name == "prompt_guard":
        return {"prompt_guard": config_dict.get("prompt_guard", {})}
    elif section_name == "tools":
        return {"tools": config_dict.get("tools", {})}
    elif section_name == "router":
        return {"router": config_dict.get("router", {})}
    else:
        return {section_name: config_dict.get(section_name, {})}


def generate_section_intent(
    base_intent: str, section_name: str, operation: str = "generate"
) -> str:
    """Generate intent for section-level example."""
    section_descriptions = {
        "decisions": "routing decisions",
        "semantic_cache": "semantic cache configuration",
        "model_config": "model configuration",
        "vllm_endpoints": "vLLM endpoints configuration",
        "prompt_guard": "PII detection and prompt guard",
        "tools": "tools auto-selection configuration",
        "router": "router options",
    }

    section_desc = section_descriptions.get(section_name, section_name)

    if operation == "generate":
        return f"Generate {section_desc} for {base_intent}"
    elif operation == "add":
        return f"Add {section_desc} to configuration"
    else:
        return f"{section_desc} configuration"


def main():
    parser = argparse.ArgumentParser(description="Generate section-level examples")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with full config examples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="raw/section_examples.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--sections",
        type=str,
        nargs="+",
        default=[
            "decisions",
            "semantic_cache",
            "model_config",
            "vllm_endpoints",
            "prompt_guard",
            "tools",
        ],
        help="Sections to extract",
    )
    parser.add_argument(
        "--max-per-section", type=int, default=20, help="Maximum examples per section"
    )

    args = parser.parse_args()

    # Load input examples
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    full_examples = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                full_examples.append(json.loads(line))

    print(f"Loaded {len(full_examples)} full config examples")

    # Generate section examples
    section_examples = []

    for section_name in args.sections:
        print(f"\nProcessing section: {section_name}")
        section_count = 0

        for example in tqdm(full_examples, desc=f"Extracting {section_name}"):
            if section_count >= args.max_per_section:
                break

            # Get config content
            config_content = example.get("full_config") or example.get("config", "")
            if not config_content:
                continue

            # Parse config
            try:
                config_dict = yaml.safe_load(config_content)
                if not config_dict:
                    continue
            except yaml.YAMLError:
                continue

            # Extract section
            section_dict = extract_section(config_dict, section_name)

            # Skip if section is empty
            if not section_dict or not section_dict.get(section_name):
                continue

            # Convert to YAML
            try:
                section_yaml = yaml.dump(
                    section_dict, default_flow_style=False, sort_keys=False
                )
            except Exception as e:
                continue

            # Validate
            if not validate_yaml_syntax(section_yaml):
                continue

            # Create section example
            section_example = {
                "id": f"section_{section_name}_{example.get('id', 'unknown')}_{section_count}",
                "config": section_yaml,
                "intent": generate_section_intent(
                    example.get("intent", "Configuration"), section_name, "generate"
                ),
                "section": section_name,
                "deployment_context": example.get("deployment_context", "unknown"),
                "source": "section",
                "base_id": example.get("id", "unknown"),
                "full_config_intent": example.get("intent", ""),
            }

            section_examples.append(section_example)
            section_count += 1

        print(f"  Generated {section_count} examples for {section_name}")

    print(f"\nTotal section examples generated: {len(section_examples)}")

    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in section_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Written to {output_path}")

    # Print statistics
    print("\nStatistics:")
    sections = {}
    for ex in section_examples:
        section = ex.get("section", "unknown")
        sections[section] = sections.get(section, 0) + 1
    print(f"  By section: {sections}")

    return 0


if __name__ == "__main__":
    exit(main())
