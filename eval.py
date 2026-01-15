#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
import torchvision.transforms as T
import pyiqa

if hasattr(Image, "Resampling"):
    _RESAMPLE = Image.Resampling.BILINEAR
else:  # Pillow < 9.1 fallback
    _RESAMPLE = Image.BILINEAR

# NTIRE 参考评分参数
_PSNR_RANGE = (0.0, 50.0)
_SSIM_RANGE = (0.5, 1.0)
_LPIPS_RANGE = (0.0, 1.0)
_WEIGHTS = {"psnr": 30.0, "ssim": 22.5, "lpips": 30.0}


def _load_tensor(
    path: str, device: torch.device, target_size: Tuple[int, int] | None = None
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """读取图像为张量，并按需调整到 target_size。"""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if target_size and img.size != target_size:
            img = img.resize(target_size, resample=_RESAMPLE)
        size = img.size
        tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    return tensor, size


def _prepare_metrics(device: torch.device) -> Dict[str, Tuple[str, Any]]:
    """
    初始化所需指标。
    返回值格式：name -> (metric_type, metric_fn)
    metric_type: "fr"（full-reference，需要 GT）、"nr"（no-reference，无需 GT）。
    """
    return {
        "psnr": ("fr", pyiqa.create_metric("psnr", device=device)),
        "ssim": ("fr", pyiqa.create_metric("ssim", device=device)),
        "lpips": ("fr", pyiqa.create_metric("lpips", device=device)),
    }


def _norm_score(score: float, index_range: Tuple[float, float]) -> float:
    """按指定区间做线性归一化。"""
    min_v, max_v = index_range
    return (score - min_v) / (max_v - min_v)


def _compute_score(psnr: float, ssim: float, lpips: float) -> float:
    """按 NTIRE 公式计算综合得分。"""
    psnr_score = _norm_score(psnr, _PSNR_RANGE)
    ssim_score = _norm_score(ssim, _SSIM_RANGE)
    lpips_score = _norm_score(1.0 - lpips / 0.4, _LPIPS_RANGE)
    return (
        _WEIGHTS["psnr"] * psnr_score
        + _WEIGHTS["ssim"] * ssim_score
        + _WEIGHTS["lpips"] * lpips_score
    )


def evaluate(
    pred_root: str, gt_root: str, device: torch.device
) -> List[Dict[str, float]]:
    """遍历预测结果并计算各项指标，按文件名匹配 GT。"""
    if not os.path.isdir(pred_root):
        raise FileNotFoundError(f"预测目录不存在：{pred_root}")
    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"GT 目录不存在：{gt_root}")

    metrics = _prepare_metrics(device)
    results: List[Dict[str, float]] = []

    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".JPG")
    pred_files = sorted(
        f for f in os.listdir(pred_root) if f.lower().endswith(image_exts)
    )

    if not pred_files:
        print("预测目录中未找到任何图像文件")
        return results

    for fname in pred_files:
        pred_path = os.path.join(pred_root, fname)

        # 在 GT 目录中用同名文件匹配；若同名不同扩展，逐一尝试
        stem, _ = os.path.splitext(fname)
        gt_path = None
        for ext in image_exts:
            candidate = os.path.join(gt_root, stem + ext)
            if os.path.isfile(candidate):
                gt_path = candidate
                break
        if gt_path is None:
            print(f"[WARN] 缺少 GT：{fname}")
            continue

        gt_tensor, gt_size = _load_tensor(gt_path, device)
        pred_tensor, _ = _load_tensor(pred_path, device, target_size=gt_size)

        with torch.no_grad():
            scores: Dict[str, float] = {}
            for name, (metric_type, metric_fn) in metrics.items():
                if metric_type == "fr":
                    scores[name] = metric_fn(pred_tensor, gt_tensor).item()

            scores["score"] = _compute_score(
                scores["psnr"], scores["ssim"], scores["lpips"]
            )

        results.append({"case": stem, **scores})
        print(
            f"{stem}: "
            f"PSNR={scores['psnr']:.4f}, "
            f"SSIM={scores['ssim']:.4f}, "
            f"LPIPS={scores['lpips']:.4f}, "
            f"SCORE={scores['score']:.4f}"
        )

    return results


def summarize(results: List[Dict[str, float]]) -> Dict[str, float] | None:
    """打印平均指标，并返回均值字典。"""
    if not results:
        print("未生成任何结果，请检查输入路径。")
        return None

    mean_scores: Dict[str, float] = {}
    for key in ("psnr", "ssim", "lpips", "score"):
        valid_values = [r[key] for r in results if key in r and r[key] is not None]
        if not valid_values:
            continue
        mean_scores[key] = sum(valid_values) / len(valid_values)

    print("\n平均指标：")
    for key, value in mean_scores.items():
        print(f"{key.upper()}: {value:.4f}")
    return mean_scores


def _pred_root_tag(pred_root: str) -> str:
    """用 pred_root 最后两个路径文件夹名称生成 tag（用下划线连接）。"""
    p = Path(pred_root).expanduser()
    parts = [x for x in p.parts if x not in (os.sep, "")]
    if not parts:
        return "pred_root"
    last = parts[-1]
    second_last = parts[-2] if len(parts) >= 2 else ""
    return f"{second_last}_{last}" if second_last else last


def save_results_txt(
    results: List[Dict[str, float]],
    mean_scores: Dict[str, float] | None,
    pred_root: str,
    out_dir: str | None = None,
) -> str:
    """将每一帧结果保存为一行文本，最后追加平均结果，写入 metric_result/*.txt。"""
    base_dir = Path(out_dir) if out_dir else (Path(__file__).resolve().parent / "metric_result")
    base_dir.mkdir(parents=True, exist_ok=True)

    tag = _pred_root_tag(pred_root)
    out_path = base_dir / f"{tag}.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("case\tpsnr\tssim\tlpips\tscore\n")
        for r in results:
            f.write(
                f"{r['case']}\t"
                f"{r['psnr']:.6f}\t"
                f"{r['ssim']:.6f}\t"
                f"{r['lpips']:.6f}\t"
                f"{r['score']:.6f}\n"
            )

        if mean_scores is None:
            f.write("MEAN\tN/A\tN/A\tN/A\tN/A\n")
        else:
            f.write(
                f"MEAN\t"
                f"{mean_scores.get('psnr', float('nan')):.6f}\t"
                f"{mean_scores.get('ssim', float('nan')):.6f}\t"
                f"{mean_scores.get('lpips', float('nan')):.6f}\t"
                f"{mean_scores.get('score', float('nan')):.6f}\n"
            )

    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算预测结果与GT的 PSNR / SSIM / LPIPS / 综合评分（基于 pyiqa）"
    )
    parser.add_argument(
        "--pred_root",
        default="FreeFusion/result_kalantari_testing",
        help="预测结果根目录（每个子目录对应一个样本）",
    )
    parser.add_argument(
        "--gt_root",
        default="FreeFusion/datasets/SICE/gt_resize",
        help="GT 图像所在目录",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="选择设备：auto / cpu / cuda(:id)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"使用设备：{device}")
    results = evaluate(args.pred_root, args.gt_root, device)
    mean_scores = summarize(results)
    out_path = save_results_txt(results, mean_scores, args.pred_root)
    print(f"\n已保存结果到：{out_path}")


if __name__ == "__main__":
    main()

