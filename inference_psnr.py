"""
C3G Inference Script — render context views and compute PSNR.

Two usage modes
---------------
1. Dataset mode (recommended):
   Reuse the existing dataset pipeline and compute PSNR on every test batch.

   python inference_psnr.py \
       +evaluation=re10k \
       mode=test \
       checkpointing.load=path/to/checkpoint.ckpt \
       wandb.mode=disabled \
       test.save_image=true \
       test.save_compare=true

2. Custom-image mode (see run_inference_custom() below):
   Pass raw image tensors and camera parameters directly.
   Useful when you have your own images outside the dataset system.
"""

import os
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import install_import_hook
from omegaconf import DictConfig

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule, get_data_shim
    from src.dataset.types import BatchedExample
    from src.evaluation.metrics import compute_psnr, compute_ssim
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.image_io import save_image
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.load_foundation_model import load_foundation_model
    from src.model.model_wrapper import ModelWrapper


# ---------------------------------------------------------------------------
# Mode 1 — dataset-driven inference (runs via `python inference_psnr.py ...`)
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    """
    Standard entry point. Mirrors src/main.py but always runs in test mode
    and prints per-batch PSNR at the end of each step.
    """
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Force test mode regardless of the config value.
    cfg.mode = "test"

    from src.misc.LocalLogger import LocalLogger

    logger = LocalLogger()

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    if checkpoint_path is None:
        raise ValueError(
            "No checkpoint specified. "
            "Pass checkpointing.load=<path> on the command line."
        )

    step_tracker = StepTracker()

    vggt, dino, lseg_feature_extractor, clip_model, feature_dim = (
        load_foundation_model(cfg)
    )
    cfg.model.encoder.feature_dim = (
        feature_dim if cfg.train.feature_rendering_loss > 0 else 0
    )

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)

    model = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        decoder,
        get_losses(cfg.loss),
        step_tracker,
        vggt=vggt,
        dino=dino,
        clip=clip_model,
        lseg_feature_extractor=lseg_feature_extractor,
        mode="test",
    )

    # Load checkpoint weights.
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=0,
    )
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    data_shim = get_data_shim(encoder)

    all_psnr = []
    output_root = Path(cfg.test.output_path) / cfg_dict.wandb.name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference on {len(test_loader)} batches …")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device.
            batch = _batch_to_device(batch, device)
            # Apply encoder data shim (e.g. normalisation).
            batch: BatchedExample = data_shim(batch)

            b, v, _, h, w = batch["target"]["image"].shape
            if h != 224 or w != 224:
                b_ctx, cv, _, ch, cw = batch["context"]["image"].shape
                batch["context"]["image"] = F.interpolate(
                    batch["context"]["image"].reshape(b_ctx * cv, 3, ch, cw),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b_ctx, cv, 3, 224, 224)

            context_feature = (
                model.forward_foundation_model(batch["context"]["image"])
                if encoder.cfg.feature_dim
                else None
            )

            # Encode context views → 3-D Gaussians.
            gaussians = encoder(
                batch["context"],
                global_step=0,
                context_feature=context_feature,
            )

            # -----------------------------------------------------------------
            # Render BOTH target views AND context views (self-reconstruction).
            # Concatenating them lets a single decoder call cover everything.
            # -----------------------------------------------------------------
            all_extrinsics = torch.cat(
                [batch["target"]["extrinsics"], batch["context"]["extrinsics"]], dim=1
            )
            all_intrinsics = torch.cat(
                [batch["target"]["intrinsics"], batch["context"]["intrinsics"]], dim=1
            )
            all_near = torch.cat(
                [batch["target"]["near"], batch["context"]["near"]], dim=1
            )
            all_far = torch.cat(
                [batch["target"]["far"], batch["context"]["far"]], dim=1
            )

            output = decoder.forward(
                gaussians,
                all_extrinsics,
                all_intrinsics,
                all_near,
                all_far,
                (h, w),
            )

            # Split rendered output back into target / context parts.
            n_target = batch["target"]["image"].shape[1]
            rendered_target = output.color[:, :n_target]        # [B, Vt, 3, H, W]
            rendered_context = output.color[:, n_target:]       # [B, Vc, 3, H, W]

            # Ground-truth for target views is already in [0, 1].
            gt_target = batch["target"]["image"]                # [B, Vt, 3, H, W]

            # Ground-truth for context views: convert from normalised [-1,1] → [0,1].
            gt_context = (batch["context"]["image"] + 1) / 2   # [B, Vc, 3, H, W]

            # -----------------------------------------------------------------
            # PSNR — computed per image, averaged over the batch.
            # -----------------------------------------------------------------
            psnr_target = compute_psnr(
                rearrange(gt_target, "b v c h w -> (b v) c h w"),
                rearrange(rendered_target, "b v c h w -> (b v) c h w"),
            )  # [(B*Vt)]

            psnr_context = compute_psnr(
                rearrange(gt_context, "b v c h w -> (b v) c h w"),
                rearrange(rendered_context, "b v c h w -> (b v) c h w"),
            )  # [(B*Vc)]

            ssim_target = compute_ssim(
                rearrange(gt_target, "b v c h w -> (b v) c h w"),
                rearrange(rendered_target, "b v c h w -> (b v) c h w"),
            )

            mean_psnr_target = psnr_target.mean().item()
            mean_psnr_ctx = psnr_context.mean().item()
            mean_ssim = ssim_target.mean().item()
            all_psnr.append(mean_psnr_target)

            scene = batch["scene"][0] if isinstance(batch["scene"], list) else batch["scene"]
            print(
                f"[{batch_idx:04d}] scene={scene}  "
                f"PSNR(target)={mean_psnr_target:.2f} dB  "
                f"PSNR(context/self)={mean_psnr_ctx:.2f} dB  "
                f"SSIM(target)={mean_ssim:.4f}"
            )

            # Save rendered images if requested.
            if cfg.test.save_image:
                scene_dir = output_root / scene
                for idx_v, color in enumerate(rendered_target[0]):
                    frame_idx = batch["target"]["index"][0][idx_v].item()
                    save_image(color, scene_dir / f"rendered_{frame_idx:06d}.png")
                for idx_v, color in enumerate(rendered_context[0]):
                    frame_idx = batch["context"]["index"][0][idx_v].item()
                    save_image(color, scene_dir / f"rendered_ctx_{frame_idx:06d}.png")

            if cfg.test.save_compare:
                from src.misc.image_io import prep_image
                from src.visualization.layout import hcat, vcat
                from src.visualization.annotation import add_label
                from src.misc.utils import inverse_normalize

                ctx_imgs = inverse_normalize(batch["context"]["image"][0])
                error = (gt_target[0] - rendered_target[0].clamp(0, 1)).abs()
                comparison = hcat(
                    add_label(vcat(*ctx_imgs), "Context"),
                    add_label(vcat(*gt_target[0]), "Target GT"),
                    add_label(vcat(*rendered_target[0]), "Rendered"),
                    add_label(vcat(*error), "Error"),
                )
                save_image(
                    comparison,
                    output_root / f"{scene}_{mean_psnr_target:.2f}dB.png",
                )

    if all_psnr:
        avg = sum(all_psnr) / len(all_psnr)
        print(f"\n{'='*60}")
        print(f"Mean PSNR (target views) over {len(all_psnr)} scenes: {avg:.2f} dB")
        print(f"Results saved to: {output_root}")


# ---------------------------------------------------------------------------
# Mode 2 — custom-image API (import and call directly)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference_custom(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    images: torch.Tensor,           # [B, V, 3, H, W], float, range [0, 1]
    extrinsics: torch.Tensor,       # [B, V, 4, 4]  world→camera
    intrinsics: torch.Tensor,       # [B, V, 3, 3]
    near: torch.Tensor,             # [B, V]
    far: torch.Tensor,              # [B, V]
    render_size: Optional[tuple] = None,
    data_shim=None,
) -> dict:
    """
    Minimal inference helper for custom images.

    Parameters
    ----------
    encoder, decoder : loaded C3G encoder and decoder modules.
    images           : RGB images in [0, 1].  Will be used as BOTH context
                       (input to the encoder) and render target (for PSNR).
    extrinsics       : camera-to-world or world-to-camera transforms — must
                       match the convention expected by your checkpoint.
    intrinsics       : 3×3 camera matrices (fx 0 cx / 0 fy cy / 0 0 1).
    near, far        : depth clipping planes.
    render_size      : (H, W) output resolution.  Defaults to input resolution.
    data_shim        : optional data-shim callable from get_data_shim(encoder).

    Returns
    -------
    dict with keys:
        "rendered"   – rendered images [B, V, 3, H, W], range [0, 1]
        "psnr"       – per-image PSNR tensor [B*V]
        "psnr_mean"  – scalar float
        "gaussians"  – the predicted Gaussians object
    """
    B, V, C, H, W = images.shape
    assert C == 3, "Expected RGB images with shape [B, V, 3, H, W]"
    h, w = render_size if render_size is not None else (H, W)

    # Build a BatchedExample compatible with the encoder.
    # Context images must be in the normalised range expected by the encoder.
    # The standard C3G pipeline normalises them to approximately [-1, 1].
    context_images_norm = images * 2.0 - 1.0  # [0,1] → [-1,1]

    batch: BatchedExample = {
        "context": {
            "image": context_images_norm,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "near": near,
            "far": far,
            "index": torch.arange(V, device=images.device).unsqueeze(0).expand(B, -1),
            "overlap": torch.zeros(B, V, device=images.device),
        },
        "target": {
            "image": images,            # [0, 1] — used only for PSNR below
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "near": near,
            "far": far,
            "index": torch.arange(V, device=images.device).unsqueeze(0).expand(B, -1),
            "overlap": torch.zeros(B, V, device=images.device),
        },
        "scene": [f"scene_{i}" for i in range(B)],
    }

    # Apply optional encoder shim (normalisation, bounds, etc.).
    if data_shim is not None:
        batch = data_shim(batch)
        # After shim the context images may have changed; keep target in [0,1].
        batch["target"]["image"] = images

    # Resize context images to 224×224 if needed (encoder requirement).
    ctx_h, ctx_w = batch["context"]["image"].shape[-2:]
    if ctx_h != 224 or ctx_w != 224:
        batch["context"]["image"] = F.interpolate(
            batch["context"]["image"].reshape(B * V, 3, ctx_h, ctx_w),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, V, 3, 224, 224)

    # Encode → Gaussians.
    gaussians = encoder(batch["context"], global_step=0)

    # Decode → rendered images.
    output = decoder.forward(
        gaussians,
        batch["target"]["extrinsics"],
        batch["target"]["intrinsics"],
        batch["target"]["near"],
        batch["target"]["far"],
        (h, w),
    )

    rendered = output.color  # [B, V, 3, H, W], already in [0, 1]

    # PSNR between rendered and original input images.
    psnr_vals = compute_psnr(
        rearrange(images.clamp(0, 1), "b v c h w -> (b v) c h w"),
        rearrange(rendered.clamp(0, 1), "b v c h w -> (b v) c h w"),
    )  # [B*V]

    return {
        "rendered": rendered,
        "psnr": psnr_vals,
        "psnr_mean": psnr_vals.mean().item(),
        "gaussians": gaussians,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_to_device(batch, device):
    """Recursively move all tensors in a nested dict/list to `device`."""
    if isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_batch_to_device(v, device) for v in batch]
        return type(batch)(moved)
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def load_model_from_checkpoint(
    config_overrides: list[str],
    checkpoint_path: str,
    device: str = "cuda",
) -> tuple:
    """
    Convenience function: load encoder + decoder from a checkpoint without
    going through the full Hydra CLI.

    Parameters
    ----------
    config_overrides : Hydra-style overrides, e.g.
        ["+evaluation=re10k", "model/encoder=vggt"]
    checkpoint_path  : path to the .ckpt file.
    device           : "cuda" or "cpu".

    Returns
    -------
    (encoder, decoder, data_shim)
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str(Path(__file__).parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_dict = compose("main", overrides=config_overrides)

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    _, _, _, _, feature_dim = load_foundation_model(cfg)
    cfg.model.encoder.feature_dim = 0  # disable feature head for plain inference

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip "encoder." / "decoder." prefixes written by ModelWrapper.
    enc_state = {
        k[len("encoder."):]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    dec_state = {
        k[len("decoder."):]: v
        for k, v in state_dict.items()
        if k.startswith("decoder.")
    }
    encoder.load_state_dict(enc_state, strict=False)
    decoder.load_state_dict(dec_state, strict=False)

    dev = torch.device(device)
    encoder.to(dev).eval()
    decoder.to(dev).eval()

    return encoder, decoder, get_data_shim(encoder)


# ---------------------------------------------------------------------------
# Quick-start example (run as a script without Hydra)
# ---------------------------------------------------------------------------

def _demo():
    """
    Minimal demo: synthesise random 'images', run inference, print PSNR.
    Replace the random tensors with real images / camera params.
    """
    checkpoint = "pretrained_weights/model.ckpt"   # <-- change this
    overrides = ["+evaluation=re10k"]               # <-- match your checkpoint

    encoder, decoder, shim = load_model_from_checkpoint(overrides, checkpoint)
    device = next(encoder.parameters()).device

    # Dummy data — replace with your own images and camera matrices.
    B, V = 1, 2
    images = torch.rand(B, V, 3, 256, 256, device=device)
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1).clone()
    intrinsics = torch.tensor(
        [[256.0, 0, 128], [0, 256, 128], [0, 0, 1]], device=device
    ).unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1).clone()
    near = torch.full((B, V), 0.1, device=device)
    far = torch.full((B, V), 100.0, device=device)

    results = run_inference_custom(
        encoder, decoder, images, extrinsics, intrinsics, near, far, data_shim=shim
    )

    print(f"Rendered shape : {results['rendered'].shape}")
    print(f"Per-image PSNR : {results['psnr'].tolist()}")
    print(f"Mean PSNR      : {results['psnr_mean']:.2f} dB")


if __name__ == "__main__":
    main()   # Hydra-based dataset-driven inference
