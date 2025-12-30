#!/usr/bin/env python3
"""
Generate synthetic config examples directly (no API needed).
Creates diverse examples for training the LLM config generator.
"""

import json
import yaml
import random
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Model configurations
MODELS = {
    "qwen3": {"reasoning_family": "qwen3"},
    "llama3": {"reasoning_family": "qwen3"},  # Can use qwen3 reasoning
    "phi4": {"reasoning_family": "qwen3"},
    "mistral": {"reasoning_family": "qwen3"},
    "deepseek": {"reasoning_family": "deepseek"},
    "gpt-oss-20b": {"reasoning_family": "gpt-oss"},
    "openai/gpt-oss-20b": {"reasoning_family": "gpt-oss"},
}

# Deployment contexts
DEPLOYMENT_CONTEXTS = [
    "quickstart",
    "ai-gateway",
    "istio",
    "kubernetes",
    "dynamic-config",
    "production-stack",
    "aibrix",
    "routing-strategies",
]

# Categories
CATEGORIES = [
    "business",
    "law",
    "psychology",
    "biology",
    "chemistry",
    "history",
    "health",
    "economics",
    "math",
    "physics",
    "computer_science",
    "philosophy",
    "engineering",
    "other",
]

# Features
FEATURES = [
    "semantic_cache",
    "pii_detection",
    "jailbreak_detection",
    "routing_decisions",
    "tools",
    "observability",
    "lora_routing",
    "response_api",
    "hallucination_mitigation",
]

# Cache backends
CACHE_BACKENDS = ["memory", "milvus", "redis", "hybrid"]

# Endpoint addresses (realistic IPs)
ENDPOINT_ADDRESSES = [
    "172.28.0.20",
    "192.168.1.100",
    "10.0.0.50",
    "172.16.0.10",
    "127.0.0.1",
    "10.10.10.20",
    "172.20.0.5",
]

ENDPOINT_PORTS = [8000, 8002, 11434, 8080, 8001]


def generate_basic_config(
    model_name: str = "qwen3",
    deployment_context: str = "quickstart",
    enable_cache: bool = True,
    enable_pii: bool = True,
    enable_tools: bool = False,
    enable_observability: bool = False,
    num_decisions: int = 3,
    complexity: str = "medium",
) -> str:
    """Generate a basic config YAML string."""

    endpoint_name = f"endpoint_{random.randint(1, 5)}"
    endpoint_address = random.choice(ENDPOINT_ADDRESSES)
    endpoint_port = random.choice(ENDPOINT_PORTS)

    model_config = MODELS.get(model_name, MODELS["qwen3"])
    reasoning_family = model_config["reasoning_family"]

    # Select random categories for decisions
    selected_categories = random.sample(CATEGORIES, min(num_decisions, len(CATEGORIES)))

    config_parts = []

    # vLLM Endpoints
    config_parts.append("vllm_endpoints:")
    config_parts.append(f'  - name: "{endpoint_name}"')
    config_parts.append(f'    address: "{endpoint_address}"')
    config_parts.append(f"    port: {endpoint_port}")
    config_parts.append(f"    weight: {random.choice([1, 2, 3])}")
    config_parts.append("")

    # Model Config
    config_parts.append("model_config:")
    config_parts.append(f'  "{model_name}":')
    config_parts.append(f'    reasoning_family: "{reasoning_family}"')
    config_parts.append(f'    preferred_endpoints: ["{endpoint_name}"]')
    config_parts.append("")

    # Categories
    config_parts.append("categories:")
    for cat in selected_categories:
        config_parts.append(f"  - name: {cat}")
        config_parts.append(f'    description: "{cat.capitalize()} related queries"')
        config_parts.append(f'    mmlu_categories: ["{cat}"]')
    config_parts.append("")

    # Strategy
    config_parts.append('strategy: "priority"')
    config_parts.append("")

    # Decisions
    config_parts.append("decisions:")
    for i, category in enumerate(selected_categories):
        priority = 100 - (i * 10)
        use_reasoning = random.choice([True, False]) if complexity == "high" else False

        config_parts.append(f'  - name: "{category}_decision"')
        config_parts.append(
            f'    description: "{category.capitalize()} related queries"'
        )
        config_parts.append(f"    priority: {priority}")
        config_parts.append("    rules:")
        config_parts.append('      operator: "AND"')
        config_parts.append("      conditions:")
        config_parts.append('        - type: "domain"')
        config_parts.append(f'          name: "{category}"')
        config_parts.append("    modelRefs:")
        config_parts.append(f'      - model: "{model_name}"')
        config_parts.append(f"        use_reasoning: {str(use_reasoning).lower()}")

        # Add plugins
        plugins = []
        plugins.append(
            {
                "type": "system_prompt",
                "configuration": {
                    "system_prompt": f"You are a {category} expert. Provide accurate and helpful responses."
                },
            }
        )

        if enable_pii:
            plugins.append(
                {
                    "type": "pii",
                    "configuration": {"enabled": True, "pii_types_allowed": []},
                }
            )

        if enable_cache and random.random() > 0.5:
            plugins.append(
                {
                    "type": "semantic-cache",
                    "configuration": {
                        "enabled": True,
                        "similarity_threshold": round(random.uniform(0.7, 0.95), 2),
                    },
                }
            )

        if plugins:
            config_parts.append("    plugins:")
            for plugin in plugins:
                config_parts.append(f"      - type: \"{plugin['type']}\"")
                config_parts.append("        configuration:")
                for key, value in plugin["configuration"].items():
                    if isinstance(value, bool):
                        config_parts.append(f"          {key}: {str(value).lower()}")
                    elif isinstance(value, (int, float)):
                        config_parts.append(f"          {key}: {value}")
                    else:
                        config_parts.append(f'          {key}: "{value}"')

        config_parts.append("")

    # Default model
    config_parts.append(f"default_model: {model_name}")
    config_parts.append("")

    # Reasoning families
    config_parts.append("reasoning_families:")
    if reasoning_family == "qwen3":
        config_parts.append("  qwen3:")
        config_parts.append('    type: "chat_template_kwargs"')
        config_parts.append('    parameter: "enable_thinking"')
    elif reasoning_family == "deepseek":
        config_parts.append("  deepseek:")
        config_parts.append('    type: "chat_template_kwargs"')
        config_parts.append('    parameter: "thinking"')
    elif reasoning_family == "gpt-oss":
        config_parts.append("  gpt-oss:")
        config_parts.append('    type: "reasoning_effort"')
        config_parts.append('    parameter: "reasoning_effort"')
    config_parts.append("")
    config_parts.append("default_reasoning_effort: high")
    config_parts.append("")

    # Semantic Cache
    if enable_cache:
        config_parts.append("semantic_cache:")
        config_parts.append("  enabled: true")
        backend = random.choice(CACHE_BACKENDS)
        config_parts.append(f'  backend_type: "{backend}"')
        config_parts.append(
            f"  similarity_threshold: {round(random.uniform(0.7, 0.9), 2)}"
        )
        config_parts.append("  max_entries: 1000")
        config_parts.append("  ttl_seconds: 3600")
        if backend == "memory":
            config_parts.append("  use_hnsw: true")
            config_parts.append("  hnsw_m: 16")
            config_parts.append("  hnsw_ef_construction: 200")
        config_parts.append("")

    # PII Detection
    if enable_pii:
        config_parts.append("classifier:")
        config_parts.append("  category_model:")
        config_parts.append('    model_id: "models/mom-domain-classifier"')
        config_parts.append("    threshold: 0.6")
        config_parts.append("    use_cpu: true")
        config_parts.append("  pii_model:")
        config_parts.append('    model_id: "models/mom-pii-classifier"')
        config_parts.append("    threshold: 0.9")
        config_parts.append("    use_cpu: true")
        config_parts.append("")

    # Tools
    if enable_tools:
        config_parts.append("tools:")
        config_parts.append("  enabled: true")
        config_parts.append("  top_k: 3")
        config_parts.append("  similarity_threshold: 0.2")
        config_parts.append('  tools_db_path: "config/tools_db.json"')
        config_parts.append("  fallback_to_empty: true")
        config_parts.append("")

    # Observability
    if enable_observability:
        config_parts.append("observability:")
        config_parts.append("  metrics:")
        config_parts.append("    enabled: true")
        config_parts.append("  tracing:")
        config_parts.append("    enabled: true")
        config_parts.append('    provider: "opentelemetry"')
        config_parts.append("    exporter:")
        config_parts.append('      type: "otlp"')
        if deployment_context == "quickstart":
            config_parts.append('      endpoint: "jaeger:4317"')
        else:
            config_parts.append('      endpoint: "jaeger-collector:4317"')
        config_parts.append("      insecure: true")
        config_parts.append("    sampling:")
        config_parts.append('      type: "always_on"')
        config_parts.append("      rate: 1.0")
        config_parts.append("    resource:")
        config_parts.append('      service_name: "vllm-semantic-router"')
        config_parts.append('      service_version: "v0.1.0"')
        config_parts.append(f'      deployment_environment: "{deployment_context}"')
        config_parts.append("")

    return "\n".join(config_parts)


def generate_intent(
    model_name: str, deployment_context: str, features: List[str], num_decisions: int
) -> str:
    """Generate a natural language intent description."""

    intent_parts = []

    # Model mention
    intent_parts.append(f"Route queries using {model_name} model")

    # Deployment context
    if deployment_context != "quickstart":
        intent_parts.append(f"in {deployment_context} deployment")

    # Features
    feature_descriptions = {
        "semantic_cache": "with semantic caching",
        "pii_detection": "with PII detection",
        "jailbreak_detection": "with jailbreak detection",
        "tools": "with tool selection",
        "observability": "with observability",
        "routing_decisions": f"with {num_decisions} routing decisions",
    }

    for feature in features:
        if feature in feature_descriptions:
            intent_parts.append(feature_descriptions[feature])

    return ", ".join(intent_parts)


def generate_example(example_id: int) -> Dict[str, Any]:
    """Generate a single example."""

    # Random configuration
    model_name = random.choice(list(MODELS.keys()))
    deployment_context = random.choice(DEPLOYMENT_CONTEXTS)
    complexity = random.choice(["low", "medium", "high"])

    # Feature flags
    enable_cache = random.random() > 0.3
    enable_pii = random.random() > 0.2
    enable_tools = random.random() > 0.6
    enable_observability = random.random() > 0.5

    # Number of decisions based on complexity
    if complexity == "low":
        num_decisions = random.randint(1, 3)
    elif complexity == "medium":
        num_decisions = random.randint(3, 7)
    else:
        num_decisions = random.randint(7, 12)

    # Collect features
    features = []
    if enable_cache:
        features.append("semantic_cache")
    if enable_pii:
        features.append("pii_detection")
    if enable_tools:
        features.append("tools")
    if enable_observability:
        features.append("observability")
    if num_decisions > 0:
        features.append("routing_decisions")

    # Generate config
    config = generate_basic_config(
        model_name=model_name,
        deployment_context=deployment_context,
        enable_cache=enable_cache,
        enable_pii=enable_pii,
        enable_tools=enable_tools,
        enable_observability=enable_observability,
        num_decisions=num_decisions,
        complexity=complexity,
    )

    # Generate intent
    intent = generate_intent(model_name, deployment_context, features, num_decisions)

    # Use case
    use_case = f"Configuration for {deployment_context} deployment with {', '.join(features[:3]) if features else 'basic routing'}"

    return {
        "id": f"ai_generated_{example_id:05d}",
        "source_file": "synthetic/ai_generated.yaml",
        "category": "main",
        "deployment_context": deployment_context,
        "intent": intent,
        "use_case": use_case,
        "complexity": complexity,
        "key_features": features,
        "full_config": config,
        "source": "ai",
        "collected_at": datetime.now().isoformat(),
    }


def main():
    """Generate synthetic examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic config examples")
    parser.add_argument(
        "--output",
        type=str,
        default="raw/synthetic_ai_examples.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num-examples", type=int, default=150, help="Number of examples to generate"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🎯 Generating {args.num_examples} synthetic examples...")
    print(f"📁 Output: {output_path}")

    examples = []
    for i in range(1, args.num_examples + 1):
        example = generate_example(i)
        examples.append(example)

        if i % 25 == 0:
            print(f"  ✅ Generated {i}/{args.num_examples} examples...")

    # Write to JSONL
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\n✅ Successfully generated {len(examples)} examples!")
    print(f"📊 Statistics:")
    print(
        f"   - Deployment contexts: {len(set(e['deployment_context'] for e in examples))}"
    )
    print(
        f"   - Models: {len(set(e['full_config'].split('model_config:')[1].split(':')[0].strip('\"') if 'model_config:' in e['full_config'] else 'unknown' for e in examples))}"
    )
    print(
        f"   - Complexity: {dict((c, sum(1 for e in examples if e['complexity'] == c)) for c in ['low', 'medium', 'high'])}"
    )


if __name__ == "__main__":
    main()
