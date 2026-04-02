"""Infer 3D Gaussians from a folder of images using C3G (pose-free mode).

Usage:
    python inference.py --image_dir /path/to/images --ckpt pretrained_weights/gaussian_decoder_multiview.ckpt
"""

import argparse
from pathlib import Path
from typing import List

import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from torchvision import transforms

from src.config import load_typed_root_config
from src.dataset.shims.normalize_shim import normalize_image
from src.model.encoder import get_encoder

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in: {folder}")
    return files


def load_posefree_encoder(ckpt_path: str, device: str):
    config_dir = str(Path(__file__).resolve().parent / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                "+training=gaussian_head_multiview",
                "mode=test",
                "wandb.mode=disabled",
            ],
        )

    cfg = load_typed_root_config(cfg_dict)
    enc_cfg = cfg.model.encoder
    assert getattr(enc_cfg, "pose_free", False), "Encoder config must have pose_free=True"

    # Ensure pure Gaussian inference path (no foundation-model features).
    enc_cfg.feature_dim = 0
    enc_cfg.gaussian_feature_dim = 0

    encoder, _ = get_encoder(enc_cfg)
    encoder = encoder.to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        state = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("encoder."):
                state[k[len("encoder."):]] = v
        missing, unexpected = encoder.load_state_dict(state, strict=False)
    elif "model" in ckpt:
        from src.misc.weight_modify import checkpoint_filter_fn

        state = checkpoint_filter_fn(ckpt["model"], encoder)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    print(f"[OK] loaded ckpt: {ckpt_path}")
    print(f"[INFO] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[INFO] missing keys (first 10): {missing[:10]}")
    return encoder


def load_images_as_context(folder: Path, max_views: int, device: str):
    img_paths = list_images(folder)
    if max_views > 0:
        img_paths = img_paths[:max_views]

    first = Image.open(img_paths[0]).convert("RGB")
    w0, h0 = first.size
    to_tensor = transforms.ToTensor()

    tensors = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        if img.size != (w0, h0):
            img = img.resize((w0, h0), Image.BILINEAR)
        t = to_tensor(img)
        t = normalize_image(t, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        tensors.append(t)

    context_image = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    return context_image, img_paths


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="C3G pose-free Gaussian inference")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing input images")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="pretrained_weights/gaussian_decoder_multiview.ckpt",
        help="Path to checkpoint",
    )
    parser.add_argument("--max_views", type=int, default=20, help="Max views to use (<=0 for all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="gaussians_from_folder.pt", help="Output .pt file")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    encoder = load_posefree_encoder(args.ckpt, device)
    context_image, used_paths = load_images_as_context(Path(args.image_dir), max_views=args.max_views, device=device)

    context = {"image": context_image}
    gaussians = encoder(context)

    out = {
        "means": gaussians.means.cpu(),
        "covariances": gaussians.covariances.cpu(),
        "harmonics": gaussians.harmonics.cpu(),
        "opacities": gaussians.opacities.cpu(),
        "feature": gaussians.feature.cpu() if gaussians.feature is not None else None,
        "image_paths": [str(p) for p in used_paths],
    }
    torch.save(out, args.out)

    print(f"\n[OK] saved: {args.out}")
    print(f"[INFO] num_views={len(used_paths)}, image_size={context_image.shape[-2:]}")
    print(f"  means:       {out['means'].shape}")
    print(f"  covariances: {out['covariances'].shape}")
    print(f"  harmonics:   {out['harmonics'].shape}")
    print(f"  opacities:   {out['opacities'].shape}")
    print(f"  feature:     {None if out['feature'] is None else out['feature'].shape}")


if __name__ == "__main__":
    main()
