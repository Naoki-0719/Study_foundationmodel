"""SAM3によるgrasp/contain/pickのIoU評価。

1枚のカラーGT画像から以下の色を抽出する。
    grasp   : RGB(255, 0, 0)   [赤]
    contain : RGB(0, 0, 255)   [青]
    pick    : RGB(0, 255, 0)   [緑]

外部のprompt.tsv / affordance_colors.tsvは使用しない。
SAM3の推論はローカルで行う（必要なモデル重みはあらかじめ用意すること）。
"""

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# SAM3のテキストプロンプト（key）とGTのRGB値（value）をここに定義
AFFORDANCE_COLORS = {
    "grasp": (255, 0, 0),
    "contain": (0, 0, 255),
    "pick": (0, 255, 0),
}

FIELDNAMES = [
    "camera_type",
    "object_name",
    "num_objects",
    "prompt_id",
    "affordance",
    "gt_color_r",
    "gt_color_g",
    "gt_color_b",
    "image",
    "iou",
    "num_pred_masks",
    "gt_pixels",
    "pred_pixels",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate 3 SAM3 affordance prompts against an RGB GT image."
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--prompt_id", default="")
    parser.add_argument("--camera_type", default="")
    parser.add_argument("--object_name", default="")
    parser.add_argument("--num_objects", type=int, default=-1)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="SAM3マスク確率の二値化閾値（0～1）。",
    )
    parser.add_argument("--output_dir", default="affordance_masks_out")
    parser.add_argument("--csv_out", default="affordance_iou_results.csv")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def find_gt_image(gt_dir, stem):
    """入力画像と同名（拡張子は任意）のGTファイルを探す。"""
    for filename in os.listdir(gt_dir):
        base, ext = os.path.splitext(filename)
        if base == stem and ext.lower() in IMAGE_EXTS:
            candidate = os.path.join(gt_dir, filename)
            if os.path.isfile(candidate):
                return candidate
    return None


def load_all_affordance_gt(path, size):
    """1枚のカラーGTからaffordanceごとのboolマスクを作成する。"""
    with Image.open(path) as gt_img:
        gt = gt_img.convert("RGB")
        if gt.size != size:
            gt = gt.resize(size, Image.Resampling.NEAREST)
        gt_np = np.asarray(gt)

    return {
        affordance: np.all(gt_np == np.asarray(color, dtype=np.uint8), axis=-1)
        for affordance, color in AFFORDANCE_COLORS.items()
    }


def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    # 元コードと同様、予測・GTとも空の場合はIoU=1.0とする。
    return 1.0 if union == 0 else float(intersection / union)


def get_merged_prediction_mask(output, gt_shape, threshold):
    """複数インスタンスのSAM3マスクをOR統合し、(H,W)のbool画像を返す。"""
    # masks_logitsは公式実装ではsigmoid後のマスク確率。
    # 存在しなければ二値マスクへフォールバックする。
    masks = output.get("masks_logits")
    if masks is None:
        masks = output["masks"]

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().float().cpu().numpy()
    else:
        masks_np = np.asarray(masks, dtype=np.float32)

    # SAM3の標準形状は(N, 1, H, W)
    if masks_np.ndim == 4:
        if masks_np.shape[1] != 1:
            raise ValueError(f"Unexpected SAM3 mask shape: {masks_np.shape}")
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    elif masks_np.ndim != 3:
        raise ValueError(f"Unexpected SAM3 mask shape: {masks_np.shape}")

    num_masks = masks_np.shape[0]
    if num_masks == 0:
        return np.zeros(gt_shape, dtype=bool), 0
    if masks_np.shape[1:] != gt_shape:
        raise ValueError(
            f"SAM3 mask size {masks_np.shape[1:]} does not match GT {gt_shape}"
        )

    return np.any(masks_np > threshold, axis=0), num_masks


def save_overlay(image_np, pred_mask, color, out_path):
    """予測領域のみ半透明で表示する。"""
    overlay = image_np.copy()
    alpha = 0.5
    blended = (
        image_np[pred_mask].astype(np.float32) * (1 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    )
    overlay[pred_mask] = blended.astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def make_row(args, affordance, color, image, iou, num_masks, gt_pixels, pred_pixels):
    return {
        "camera_type": args.camera_type,
        "object_name": args.object_name,
        "num_objects": args.num_objects,
        "prompt_id": args.prompt_id,
        "affordance": affordance,
        "gt_color_r": color[0],
        "gt_color_g": color[1],
        "gt_color_b": color[2],
        "image": image,
        "iou": iou,
        "num_pred_masks": num_masks,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # モデルの読み込みは1回のみ（ローカル重みを事前に準備すること）。
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    image_names = sorted(
        filename for filename in os.listdir(args.image_dir)
        if filename.lower().endswith(IMAGE_EXTS)
        and os.path.isfile(os.path.join(args.image_dir, filename))
    )
    rows = []
    num_evaluated_images = 0

    # CPUへの推論結果転送以外は推論用コンテキストで実行
    with torch.inference_mode():
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else torch.autocast("cpu", enabled=False)
        )
        with autocast_context:
            for image_name in image_names:
                stem = os.path.splitext(image_name)[0]
                gt_path = find_gt_image(args.gt_dir, stem)
                if gt_path is None:
                    print(f"[skip] GTがありません: {stem}")
                    continue

                with Image.open(os.path.join(args.image_dir, image_name)) as img:
                    image = img.convert("RGB")
                image_np = np.asarray(image)

                # GT画像の読み込みは画像につき1回だけ。
                gt_masks = load_all_affordance_gt(gt_path, image.size)

                # 画像特徴の抽出は画像につき1回だけ。
                state = processor.set_image(image)
                num_evaluated_images += 1

                for affordance, color in AFFORDANCE_COLORS.items():
                    # 前のプロンプトの状態を残さず同じ画像特徴を再利用する。
                    processor.reset_all_prompts(state)
                    output = processor.set_text_prompt(
                        state=state, prompt=affordance
                    )
                    gt_mask = gt_masks[affordance]
                    pred_mask, num_masks = get_merged_prediction_mask(
                        output, gt_mask.shape, args.threshold
                    )
                    iou = calculate_iou(pred_mask, gt_mask)

                    rows.append(make_row(
                        args, affordance, color, image_name, iou, num_masks,
                        int(gt_mask.sum()), int(pred_mask.sum())
                    ))
                    print(
                        f"{image_name} / {affordance}: "
                        f"IoU={iou:.4f}, pred_masks={num_masks}"
                    )
                    save_overlay(
                        image_np, pred_mask, color,
                        os.path.join(args.output_dir, f"{stem}_{affordance}_overlay.png"),
                    )
                    # 個別予測マスクも保存（白=予測、黒=背景）。
                    Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(
                        os.path.join(args.output_dir, f"{stem}_{affordance}_mask.png")
                    )

    # 3クラスそれぞれの平均IoU
    mean_by_affordance = {}
    for affordance, color in AFFORDANCE_COLORS.items():
        ious = [r["iou"] for r in rows if r["affordance"] == affordance]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_by_affordance[affordance] = mean_iou
        rows.append(make_row(
            args, affordance, color, "MEAN", mean_iou, "", "", ""
        ))

    # 3クラス平均（画像ごとのIoUをクラス内平均した後にマクロ平均）。
    macro_mean_iou = float(np.mean(list(mean_by_affordance.values())))
    if num_evaluated_images == 0:
        macro_mean_iou = 0.0
    rows.append(make_row(
        args, "ALL", ("", "", ""), "MEAN_ALL", macro_mean_iou, "", "", ""
    ))

    with open(args.csv_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n評価画像数: {num_evaluated_images}")
    for affordance, mean_iou in mean_by_affordance.items():
        print(f"{affordance} 平均IoU: {mean_iou:.4f}")
    print(f"全クラス平均IoU: {macro_mean_iou:.4f}")
    print(f"CSV出力先: {args.csv_out}")


if __name__ == "__main__":
    main()
