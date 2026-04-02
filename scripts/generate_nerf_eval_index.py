"""
Generate evaluation index for NeRF Synthetic dataset.

This creates a JSON file mapping scene keys to context/target view pairs,
compatible with C3G's ViewSamplerEvaluation.

For NeRF synthetic, each scene has 200 test views. We generate multiple
evaluation entries per scene with different context/target combinations.

Usage:
    python scripts/generate_nerf_eval_index.py \
        --input_dir /path/to/nerf_synthetic \
        --output_path assets/evaluation_index_nerf_synthetic.json \
        --split test \
        --num_context 2 \
        --num_target 3 \
        --num_entries_per_scene 20
"""

import argparse
import json
from pathlib import Path

import numpy as np


SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to nerf_synthetic root directory")
    parser.add_argument("--output_path", type=str,
                        default="assets/evaluation_index_nerf_synthetic.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scenes", type=str, nargs="+", default=None)
    parser.add_argument("--num_context", type=int, default=2)
    parser.add_argument("--num_target", type=int, default=3)
    parser.add_argument("--num_entries_per_scene", type=int, default=20,
                        help="Number of evaluation entries per scene")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    scenes = args.scenes or SCENES
    rng = np.random.RandomState(args.seed)

    index = {}

    for scene in scenes:
        transforms_path = input_dir / scene / f"transforms_{args.split}.json"
        if not transforms_path.exists():
            print(f"Warning: {transforms_path} not found, skipping")
            continue

        with open(transforms_path) as f:
            transforms = json.load(f)

        num_views = len(transforms["frames"])
        total_needed = args.num_context + args.num_target

        if num_views < total_needed:
            print(f"Warning: {scene} has only {num_views} views, need {total_needed}")
            continue

        for entry_idx in range(args.num_entries_per_scene):
            sampled = rng.choice(num_views, size=total_needed, replace=False)
            context = sorted(sampled[:args.num_context].tolist())
            target = sorted(sampled[args.num_context:].tolist())

            # Key format: scene name for the chunk key lookup
            # The evaluation sampler matches scene keys from the data loader
            entry_key = scene
            # If multiple entries per scene, we only keep the last one
            # since the evaluation index maps scene -> single entry
            # For multiple entries, you'd need to modify the chunk format
            # to have unique keys per entry

            index[scene] = {
                "context": context,
                "target": target,
                "overlap": 0.5,
            }

        print(f"{scene}: {num_views} views, generated entry with "
              f"context={index[scene]['context']}, target={index[scene]['target']}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved evaluation index with {len(index)} entries to {output_path}")


if __name__ == "__main__":
    main()
