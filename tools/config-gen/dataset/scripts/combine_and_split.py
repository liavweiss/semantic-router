#!/usr/bin/env python3
"""
Combine all examples and split into train/validation/test sets.

This script combines examples from multiple sources, deduplicates them,
and splits into training sets.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from collections import defaultdict


def load_examples(input_files: List[str]) -> List[Dict[str, Any]]:
    """Load examples from multiple JSONL files."""
    all_examples = []

    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}")
            continue

        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    all_examples.append(json.loads(line))

    return all_examples


def deduplicate_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate examples based on config content."""
    seen_configs = set()
    unique_examples = []

    for example in examples:
        config = example.get("config") or example.get("full_config", "")
        config_hash = hash(config)

        if config_hash not in seen_configs:
            seen_configs.add(config_hash)
            unique_examples.append(example)

    return unique_examples


def split_examples(
    examples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split examples into train/val/test sets.

    Args:
        examples: List of examples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility

    Returns:
        (train_examples, val_examples, test_examples)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Shuffle with seed
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Calculate split indices
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_examples = shuffled[:train_end]
    val_examples = shuffled[train_end:val_end]
    test_examples = shuffled[val_end:]

    return train_examples, val_examples, test_examples


def generate_statistics(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate dataset statistics."""
    stats = {
        "total": len(examples),
        "by_source": defaultdict(int),
        "by_deployment_context": defaultdict(int),
        "by_complexity": defaultdict(int),
    }

    for example in examples:
        source = example.get("source", "unknown")
        stats["by_source"][source] += 1

        context = example.get("deployment_context", "unknown")
        stats["by_deployment_context"][context] += 1

        complexity = example.get("complexity", "unknown")
        stats["by_complexity"][complexity] += 1

    # Convert defaultdicts to regular dicts
    stats["by_source"] = dict(stats["by_source"])
    stats["by_deployment_context"] = dict(stats["by_deployment_context"])
    stats["by_complexity"] = dict(stats["by_complexity"])

    return stats


def convert_to_training_format(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert example to training format (instruction/input/output).

    Args:
        example: Original example dict

    Returns:
        Training format dict
    """
    intent = example.get("intent", "Generate semantic-router configuration")
    config = example.get("config") or example.get("full_config", "")

    return {
        "instruction": intent,
        "input": "",
        "output": config,
    }


def main():
    parser = argparse.ArgumentParser(description="Combine and split examples")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input JSONL file(s) to combine",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed",
        help="Output directory for splits",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )
    parser.add_argument(
        "--deduplicate", action="store_true", help="Remove duplicate examples"
    )
    parser.add_argument(
        "--training-format",
        action="store_true",
        help="Convert to training format (instruction/input/output)",
    )

    args = parser.parse_args()

    # Load examples
    print("Loading examples...")
    all_examples = load_examples(args.input)
    print(f"Loaded {len(all_examples)} examples")

    # Deduplicate if requested
    if args.deduplicate:
        print("Deduplicating...")
        before_count = len(all_examples)
        all_examples = deduplicate_examples(all_examples)
        after_count = len(all_examples)
        print(f"Removed {before_count - after_count} duplicates ({after_count} unique)")

    # Generate statistics
    print("\nGenerating statistics...")
    stats = generate_statistics(all_examples)
    print(f"Total examples: {stats['total']}")
    print(f"By source: {stats['by_source']}")
    print(f"By deployment context: {stats['by_deployment_context']}")
    print(f"By complexity: {stats['by_complexity']}")

    # Split
    print(
        f"\nSplitting into train/val/test ({args.train_ratio}/{args.val_ratio}/{args.test_ratio})..."
    )
    train_examples, val_examples, test_examples = split_examples(
        all_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Train: {len(train_examples)}")
    print(f"Validation: {len(val_examples)}")
    print(f"Test: {len(test_examples)}")

    # Convert to training format if requested
    if args.training_format:
        print("\nConverting to training format...")
        train_examples = [convert_to_training_format(ex) for ex in train_examples]
        val_examples = [convert_to_training_format(ex) for ex in val_examples]
        test_examples = [convert_to_training_format(ex) for ex in test_examples]

    # Write splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting splits to {output_dir}...")

    # Write train
    with open(output_dir / "train.jsonl", "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    # Write validation
    with open(output_dir / "val.jsonl", "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")

    # Write test
    with open(output_dir / "test.jsonl", "w") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")

    # Write statistics
    stats["splits"] = {
        "train": len(train_examples),
        "val": len(val_examples),
        "test": len(test_examples),
    }

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Files written:")
    print(f"  - {output_dir / 'train.jsonl'}")
    print(f"  - {output_dir / 'val.jsonl'}")
    print(f"  - {output_dir / 'test.jsonl'}")
    print(f"  - {output_dir / 'statistics.json'}")

    return 0


if __name__ == "__main__":
    exit(main())
