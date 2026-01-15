import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from basicsr.archs.uformer_arch import Uformer
from basicsr.archs.Restormer_arch import Restormer
from basicsr.utils import imwrite as save_img
from skimage import img_as_ubyte
from tqdm import tqdm
import math


try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def _extract_index(path: str) -> int:
    """Extract numeric index from filename to ensure stable ordering."""
    import re

    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def _safe_index(paths, idx):
    """Clamp index into valid range (supports negative indices)."""
    if idx < 0:
        idx = len(paths) + idx
    idx = max(0, min(idx, len(paths) - 1))
    return paths[idx]


def load_sequence_tensor(seq_dir: str):
    """Read five-frame LDR sequence and stack to shape (1, 15, H, W)."""
    img_paths = sorted(
        glob(os.path.join(seq_dir, "*.*")),
        key=_extract_index,
    )
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {seq_dir}")

    # Follow the training order: darker_min, frame_min, middle, frame_max, brighter_max
    order_indices = [0, 1, (len(img_paths) - 1) // 2, -2, -1]
    ordered_paths = [_safe_index(img_paths, i) for i in order_indices]

    frames = []
    base_h, base_w = None, None
    for p in ordered_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Fail to read image: {p}")
        if base_h is None:
            base_h, base_w = img.shape[:2]
        elif img.shape[:2] != (base_h, base_w):
            img = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        frames.append(img)

    stacked = torch.cat(frames, dim=0).unsqueeze(0)  # (1, 15, H, W)
    return stacked, ordered_paths


def _lcm(a: int, b: int) -> int:
    if a == 0:
        return b
    if b == 0:
        return a
    return abs(a * b) // math.gcd(a, b)


def pad_to_factor(x: torch.Tensor, factor: int, pad_multiple: int):
    """Pad tensor to square; each side divisible by pad_multiple."""
    h, w = x.shape[2], x.shape[3]
    target_side = int(math.ceil(max(h, w) / pad_multiple) * pad_multiple)
    padh = target_side - h
    padw = target_side - w
    if padh == 0 and padw == 0:
        return x, (0, 0)
    x = F.pad(x, (0, padw, 0, padh), mode="reflect")
    return x, (padh, padw)


def crop_inference(model, inp, factor=16, block_size=512, shave=32, win_size=8, pad_multiple=None):
    """Block-wise inference with overlap; compatible with DataParallel."""
    if pad_multiple is None:
        pad_multiple = _lcm(factor, win_size)
    if block_size <= 0:
        with torch.no_grad():
            return model(inp)

    _, _, H, W = inp.shape
    device = inp.device
    output = torch.zeros((1, 3, H, W), device=device)
    weight = torch.zeros_like(output)

    y_starts = [0] if H <= block_size else list(range(0, H - block_size + 1, block_size))
    if y_starts[-1] != H - block_size:
        y_starts.append(H - block_size)

    x_starts = [0] if W <= block_size else list(range(0, W - block_size + 1, block_size))
    if x_starts[-1] != W - block_size:
        x_starts.append(W - block_size)

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + block_size, H)
            x1 = min(x0 + block_size, W)

            top = max(0, y0 - shave)
            bottom = min(H, y1 + shave)
            left = max(0, x0 - shave)
            right = min(W, x1 + shave)

            patch = inp[:, :, top:bottom, left:right]
            hp, wp = patch.shape[2], patch.shape[3]
            target_side = int(math.ceil(max(hp, wp) / pad_multiple) * pad_multiple)
            Hp = target_side
            Wp = target_side
            padh = Hp - hp
            padw = Wp - wp
            patch = F.pad(patch, (0, padw, 0, padh), mode="reflect")

            with torch.no_grad():
                out_patch = model(patch)
            out_patch = out_patch[:, :, :hp, :wp]

            block_h = y1 - y0
            block_w = x1 - x0
            valid_y1 = y0 - top
            valid_y2 = valid_y1 + block_h
            valid_x1 = x0 - left
            valid_x2 = valid_x1 + block_w

            out_slice = out_patch[:, :, valid_y1:valid_y2, valid_x1:valid_x2]
            output[:, :, y0:y1, x0:x1] += out_slice
            weight[:, :, y0:y1, x0:x1] += 1

    return output / weight.clamp(min=1e-8)


def build_model(opt_path: str, weights_path: str, device: torch.device):
    with open(opt_path, "r") as f:
        cfg = yaml.load(f, Loader=Loader)

    net_cfg = cfg.get("network_g", {}).copy()
    net_cfg.pop("type", None)
    # model = Restormer(**net_cfg)
    model = Uformer(**net_cfg)
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def main():
    parser = argparse.ArgumentParser(description="Uformer HDR inference with cropping.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/notebook/data/personal/S9059954/HDR-Competition/datasets/test-stage2-wo-gt",
        help="Root folder that contains N sub-folders, each with 5 images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/notebook/data/personal/S9059954/HDR-Competition/results/uformer",
        help="Where to save fused HDR pngs.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/notebook/data/personal/S9059954/HDR-Competition/experiments/Uformer_archived_20260113_035448/models/net_g_13.pth",
        help="Path to trained Uformer checkpoint.",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="/home/notebook/data/personal/S9059954/HDR-Competition/options/Uformer.yml",
        help="YAML config used for training (to build the model).",
    )
    parser.add_argument("--crop_size", type=int, default=256, help="Crop size for block inference.")
    parser.add_argument("--crop_shave", type=int, default=32, help="Overlap size to blend tiles.")
    parser.add_argument("--factor", type=int, default=16, help="Pad factor to align with network stride.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    model = build_model(args.opt, args.weights, device)
    # get window size and encoder depth to determine valid pad multiple
    win_size = getattr(model, "win_size", None)
    if win_size is None and hasattr(model, "module"):
        win_size = getattr(model.module, "win_size", None)
    if win_size is None:
        win_size = 8

    num_enc_layers = getattr(model, "num_enc_layers", None)
    if num_enc_layers is None and hasattr(model, "module"):
        num_enc_layers = getattr(model.module, "num_enc_layers", None)
    if num_enc_layers is None:
        num_enc_layers = 4  # Uformer default encoder depth
    down_scale = 2 ** num_enc_layers
    pad_multiple = _lcm(args.factor, win_size * down_scale)

    seq_dirs = [
        d for d in sorted(glob(os.path.join(args.input_dir, "*"))) if os.path.isdir(d)
    ]
    if len(seq_dirs) == 0:
        raise RuntimeError(f"No sub-folders found under {args.input_dir}")

    for seq_dir in tqdm(seq_dirs, desc="Infer"):
        inp, _ = load_sequence_tensor(seq_dir)
        inp = inp.to(device)
        h, w = inp.shape[2], inp.shape[3]
        inp, (padh, padw) = pad_to_factor(inp, args.factor, pad_multiple)

        restored = crop_inference(
            model,
            inp,
            factor=args.factor,
            block_size=args.crop_size,
            shave=args.crop_shave,
            win_size=win_size,
            pad_multiple=pad_multiple,
        )
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

        save_path = os.path.join(args.output_dir, f"{os.path.basename(seq_dir)}.png")
        save_img(img_as_ubyte(restored), save_path)

    print(f"Inference done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
