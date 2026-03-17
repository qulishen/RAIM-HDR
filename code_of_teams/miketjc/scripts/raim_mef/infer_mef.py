#!/usr/bin/env python3
"""
Inference script for RAIM MEF challenge.
Loads 5-exposure sequences, concatenates to 15-channel input,
runs through TimeDiffiT, saves fused output as JPEG.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from models.archs.TimeDiffiT_ResNet_color_128_arch import TimeDiffiT_ResNet_color_128


def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


def load_model(model_path, in_channels, device):
    """Load TimeDiffiT model with multi-channel support."""
    # Create args-like object for model init
    class Args:
        pass
    args_obj = Args()
    args_obj.in_channels = in_channels

    model = TimeDiffiT_ResNet_color_128(dim=args_obj)

    state_dict = torch.load(model_path, map_location=device)
    clean_state = OrderedDict()
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        clean_state[key] = v

    # Adapt init_conv weights: 3 → in_channels
    if 'init_conv.weight' in clean_state:
        old_w = clean_state['init_conv.weight']
        out_c, old_in, kH, kW = old_w.shape
        if old_in != in_channels:
            repeats = (in_channels // old_in) + 1
            new_w = old_w.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
            new_w = new_w / repeats
            clean_state['init_conv.weight'] = new_w

    # Init skip_proj to zeros if not in checkpoint
    if 'skip_proj.weight' not in clean_state and in_channels != 3:
        clean_state['skip_proj.weight'] = torch.zeros(3, in_channels, 1, 1)

    model.load_state_dict(clean_state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    return model


def tiled_inference(model, img_tensor, tile_size=512, stride=256,
                    auto_normalize=True, model_time_conditioning=True):
    """Run tiled inference with overlap blending."""
    _, c, h, w = img_tensor.shape
    device = img_tensor.device

    # Pad to be divisible by tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    _, _, ph, pw = img_tensor.shape
    output = torch.zeros(1, 3, ph, pw, device=device)
    weight = torch.zeros(1, 1, ph, pw, device=device)

    # Generate tile positions
    h_positions = list(range(0, ph - tile_size + 1, stride))
    w_positions = list(range(0, pw - tile_size + 1, stride))
    if ph - tile_size not in h_positions:
        h_positions.append(ph - tile_size)
    if pw - tile_size not in w_positions:
        w_positions.append(pw - tile_size)

    for hi in h_positions:
        for wi in w_positions:
            tile = img_tensor[:, :, hi:hi+tile_size, wi:wi+tile_size]

            if auto_normalize:
                tile = normalize_to_neg_one_to_one(tile)

            if model_time_conditioning:
                # Estimate noise level from middle exposure (channels 6:9)
                mid_tile = tile[:, 6:9, :, :]
                sigma = mid_tile.std().item()
                sigma = max(sigma, 0.01)
                time_tensor = torch.tensor([sigma], device=device).float()
            else:
                time_tensor = torch.tensor([0.05], device=device).float()

            with torch.no_grad():
                pred = model(tile, time_tensor)

            if auto_normalize:
                pred = unnormalize_to_zero_to_one(pred)

            pred = pred.clamp(0, 1)
            output[:, :, hi:hi+tile_size, wi:wi+tile_size] += pred
            weight[:, :, hi:hi+tile_size, wi:wi+tile_size] += 1

    output = output / weight.clamp(min=1)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        output = output[:, :, :h, :w]

    return output


def ensemble_inference(model, img_tensor, tile_size=512, stride=256,
                       auto_normalize=True, model_time_conditioning=True):
    """Geometric self-ensemble: 8 transforms (4 rotations × 2 flips)."""
    predictions = []

    for flip in [False, True]:
        for rot in range(4):
            x = img_tensor.clone()

            # Apply transform
            if flip:
                x = torch.flip(x, [3])  # horizontal flip
            if rot > 0:
                x = torch.rot90(x, rot, [2, 3])

            # Inference
            pred = tiled_inference(model, x, tile_size, stride,
                                   auto_normalize, model_time_conditioning)

            # Inverse transform
            if rot > 0:
                pred = torch.rot90(pred, 4 - rot, [2, 3])
            if flip:
                pred = torch.flip(pred, [3])

            predictions.append(pred)

    return torch.stack(predictions).mean(dim=0)


def load_scene_exposures(scene_dir, num_exposures=5):
    """Load exposure images from a scene directory.
    Returns tensor of shape [1, num_exposures*3, H, W] in [0, 1]."""
    frames = []
    for i in range(num_exposures):
        path = os.path.join(scene_dir, f'{i}.jpg')
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    # Concatenate: [num_exp*3, H, W]
    stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
    tensor = torch.from_numpy(stacked.astype(np.float32)) / 255.0
    return tensor.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='RAIM MEF Inference')
    parser.add_argument('--model_path', required=True, help='Path to checkpoint')
    parser.add_argument('--input_dir', required=True, help='Directory with scene subdirs')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--in_channels', default=15, type=int)
    parser.add_argument('--num_exposures', default=5, type=int)
    parser.add_argument('--tile_size', default=512, type=int)
    parser.add_argument('--stride', default=256, type=int)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--jpeg_quality', default=95, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.in_channels, device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get scene directories
    scenes = sorted([d for d in os.listdir(args.input_dir)
                     if os.path.isdir(os.path.join(args.input_dir, d))])
    print(f"Found {len(scenes)} scenes")

    for i, scene_name in enumerate(scenes):
        scene_dir = os.path.join(args.input_dir, scene_name)

        # Load multi-exposure input
        img_tensor = load_scene_exposures(scene_dir, args.num_exposures)
        img_tensor = img_tensor.to(device)

        # Inference
        if args.ensemble:
            output = ensemble_inference(model, img_tensor, args.tile_size, args.stride)
        else:
            output = tiled_inference(model, img_tensor, args.tile_size, args.stride)

        # Convert to numpy
        output = output.squeeze(0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = output.transpose(1, 2, 0)  # CHW → HWC
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Save as JPEG with scene name (zero-padded 3 digits)
        out_name = f"{scene_name}.jpg"
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, output, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        print(f"[{i+1}/{len(scenes)}] {scene_name} → {out_name}")

    print(f"\nDone. {len(scenes)} images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
