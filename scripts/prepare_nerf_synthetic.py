"""
Convert NeRF Synthetic (Blender) dataset to C3G .torch chunk format.

NeRF Synthetic dataset structure:
    nerf_synthetic/
        <scene>/              # chair, drums, ficus, hotdog, lego, materials, mic, ship
            transforms_train.json
            transforms_val.json
            transforms_test.json
            train/
                r_0.png
                r_1.png
                ...
            val/
                r_0.png
                ...
            test/
                r_0.png
                ...

Each transforms_*.json contains:
    {
        "camera_angle_x": float,  # horizontal FOV in radians
        "frames": [
            {
                "file_path": "./train/r_0",   # relative path (no .png extension)
                "rotation": float,
                "transform_matrix": [[4x4 C2W matrix]]
            },
            ...
        ]
    }

Output format (C3G .torch chunks):
    datasets/nerf_synthetic/
        test/
            index.json          # {scene_key: chunk_filename}
            chunk_0.torch       # [{key, cameras, images}, ...]

Usage:
    python scripts/prepare_nerf_synthetic.py \
        --input_dir /path/to/nerf_synthetic \
        --output_dir datasets/nerf_synthetic \
        --split test
"""

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

# NeRF synthetic images are 800x800
IMAGE_SIZE = 800


def load_transforms(scene_dir: Path, split: str) -> dict:
    """Load transforms_<split>.json for a scene."""
    transforms_path = scene_dir / f"transforms_{split}.json"
    with open(transforms_path, "r") as f:
        return json.load(f)


def c2w_to_w2c_flat(c2w: np.ndarray) -> np.ndarray:
    """Convert 4x4 C2W matrix to flattened 12-element W2C (3x4)."""
    w2c = np.linalg.inv(c2w)
    return w2c[:3, :].flatten()  # 12 elements


def make_camera_18(
    fx: float, fy: float, cx: float, cy: float, c2w: np.ndarray
) -> np.ndarray:
    """
    Create 18-element camera vector matching C3G's RE10K format.

    Format: [fx, fy, cx, cy, 0, 0, r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]

    All intrinsics are normalized to [0, 1] (divided by image dimensions).
    The last 12 elements are the flattened 3x4 W2C matrix.
    """
    w2c_flat = c2w_to_w2c_flat(c2w)
    cam = np.zeros(18, dtype=np.float32)
    cam[0] = fx   # normalized fx
    cam[1] = fy   # normalized fy
    cam[2] = cx   # normalized cx
    cam[3] = cy   # normalized cy
    # cam[4], cam[5] are unused (kept as 0)
    cam[6:] = w2c_flat
    return cam


def image_to_bytes_tensor(image_path: Path, background_color=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Load an RGBA PNG image, composite onto white background, and convert
    to bytes tensor (same format as RE10K dataset).
    """
    img = Image.open(image_path).convert("RGBA")

    # Composite RGBA onto white background (NeRF synthetic convention)
    bg = Image.new("RGBA", img.size, (
        int(background_color[0] * 255),
        int(background_color[1] * 255),
        int(background_color[2] * 255),
        255,
    ))
    composited = Image.alpha_composite(bg, img).convert("RGB")

    # Encode as PNG bytes and wrap in tensor (matches RE10K chunk format)
    buffer = BytesIO()
    composited.save(buffer, format="PNG")
    return torch.tensor(np.frombuffer(buffer.getvalue(), dtype=np.uint8))


def convert_scene(
    scene_dir: Path, scene_name: str, split: str, background_color=(1.0, 1.0, 1.0)
) -> dict:
    """
    Convert a single NeRF Synthetic scene to C3G chunk format.

    Returns a dict with:
        key: str (scene name)
        cameras: Tensor [num_views, 18]
        images: list[Tensor] (PNG bytes per view)
    """
    transforms = load_transforms(scene_dir, split)
    camera_angle_x = transforms["camera_angle_x"]

    # Compute normalized intrinsics from FOV
    # fx_pixels = W / (2 * tan(fov_x / 2))
    # Normalized: fx = fx_pixels / W = 1 / (2 * tan(fov_x / 2))
    fx_norm = 1.0 / (2.0 * np.tan(camera_angle_x / 2.0))
    fy_norm = fx_norm  # Square images, same FOV in both directions
    cx_norm = 0.5      # Principal point at image center
    cy_norm = 0.5

    cameras_list = []
    images_list = []

    frames = transforms["frames"]
    print(f"  Processing {scene_name}/{split}: {len(frames)} frames")

    for frame in frames:
        # Load C2W transform matrix
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        # NeRF synthetic uses OpenGL convention (Y up, -Z forward)
        # C3G/RE10K uses OpenCV convention (Y down, Z forward)
        # Convert: flip Y and Z axes
        c2w[:, 1] *= -1  # flip Y
        c2w[:, 2] *= -1  # flip Z

        cam_18 = make_camera_18(fx_norm, fy_norm, cx_norm, cy_norm, c2w)
        cameras_list.append(cam_18)

        # Load image
        file_path = frame["file_path"]
        # Handle paths with or without extension
        if not file_path.endswith(".png"):
            file_path += ".png"
        image_path = scene_dir / file_path
        images_list.append(image_to_bytes_tensor(image_path, background_color))

    cameras = torch.tensor(np.stack(cameras_list, axis=0))

    return {
        "key": scene_name,
        "cameras": cameras,
        "images": images_list,
    }


def generate_evaluation_index(
    scenes: list[str],
    num_views_per_scene: int,
    num_context: int = 2,
    num_target: int = 3,
    seed: int = 42,
) -> dict:
    """
    Generate evaluation index JSON for NeRF Synthetic scenes.

    For each scene, picks random context/target view pairs.
    We generate multiple evaluation entries per scene to cover different viewpoints.
    """
    rng = np.random.RandomState(seed)
    index = {}

    for scene in scenes:
        all_indices = list(range(num_views_per_scene))

        # Generate multiple evaluation entries per scene
        num_entries = min(10, num_views_per_scene // (num_context + num_target))
        for entry_idx in range(num_entries):
            # Sample context and target indices without replacement
            sampled = rng.choice(all_indices, size=num_context + num_target, replace=False)
            context = sorted(sampled[:num_context].tolist())
            target = sorted(sampled[num_context:].tolist())

            entry_key = f"{scene}_{entry_idx:03d}" if num_entries > 1 else scene
            index[entry_key] = {
                "context": context,
                "target": target,
                "overlap": 0.5,  # default overlap for synthetic scenes
            }

    return index


def main():
    parser = argparse.ArgumentParser(description="Convert NeRF Synthetic to C3G format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to nerf_synthetic root directory")
    parser.add_argument("--output_dir", type=str, default="datasets/nerf_synthetic",
                        help="Output directory for C3G format data")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to convert")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="Specific scenes to convert (default: all 8 scenes)")
    parser.add_argument("--background_color", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Background color for RGBA compositing (default: white)")
    parser.add_argument("--num_context", type=int, default=2,
                        help="Number of context views for evaluation")
    parser.add_argument("--num_target", type=int, default=3,
                        help="Number of target views per evaluation entry")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation index generation")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    scenes = args.scenes or SCENES
    background_color = tuple(args.background_color)

    # Validate input directory
    for scene in scenes:
        scene_dir = input_dir / scene
        if not scene_dir.exists():
            print(f"Warning: Scene directory {scene_dir} not found, skipping")
            scenes = [s for s in scenes if s != scene]

    if not scenes:
        print("Error: No valid scenes found!")
        return

    # Create output directory
    split_dir = output_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Convert scenes
    # For NeRF synthetic, we store each scene as one chunk (they're small enough)
    index = {}
    all_num_views = {}

    for i, scene in enumerate(scenes):
        scene_dir = input_dir / scene
        print(f"Converting scene [{i+1}/{len(scenes)}]: {scene}")

        example = convert_scene(scene_dir, scene, args.split, background_color)
        all_num_views[scene] = len(example["images"])

        # Save as individual chunk file
        chunk_filename = f"chunk_{i}.torch"
        chunk_path = split_dir / chunk_filename
        torch.save([example], chunk_path)
        print(f"  Saved {chunk_path} ({len(example['images'])} views)")

        index[scene] = chunk_filename

    # Save index.json
    index_path = split_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved index: {index_path}")

    # Generate and save evaluation index
    # For multi-entry evaluation, we expand scene keys
    eval_index = {}
    for scene in scenes:
        num_views = all_num_views[scene]
        scene_eval = generate_evaluation_index(
            [scene], num_views,
            num_context=args.num_context,
            num_target=args.num_target,
            seed=args.seed,
        )
        eval_index.update(scene_eval)

    eval_index_path = output_dir / f"evaluation_index_nerf_synthetic.json"
    with open(eval_index_path, "w") as f:
        json.dump(eval_index, f, indent=2)
    print(f"Saved evaluation index: {eval_index_path}")
    print(f"\nDone! Converted {len(scenes)} scenes with total entries: {len(eval_index)}")
    print(f"\nTo evaluate, run:")
    print(f"  python -m src.main +evaluation=nerf_synthetic mode=test \\")
    print(f"    dataset/view_sampler@dataset.nerf_synthetic.view_sampler=evaluation \\")
    print(f"    dataset.nerf_synthetic.view_sampler.index_path={eval_index_path} \\")
    print(f"    checkpointing.load=<checkpoint_path>")


if __name__ == "__main__":
    main()
