import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate text-prompted SAM3 affordance segmentation "
            "against color-coded affordance GT masks using IoU."
        )
    )

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--gt_dir", required=True)

    parser.add_argument(
        "--affordance",
        required=True,
        help=(
            "SAM3へのテキストプロンプトとして使用し、"
            "affordance_colors.tsvのGT色参照にも使用する"
        ),
    )

    parser.add_argument(
        "--affordance_colors",
        default="affordance_colors.tsv",
    )

    parser.add_argument("--prompt_id", default="")
    parser.add_argument("--camera_type", default="")
    parser.add_argument("--object_name", default="")
    parser.add_argument("--num_objects", type=int, default=-1)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output_dir",
        default="affordance_masks_out",
    )

    parser.add_argument(
        "--csv_out",
        default="affordance_iou_results.csv",
    )

    return parser.parse_args()


def load_affordance_colors(path):
    """
    affordance_colors.tsv を読み込む。

    形式:
        affordance    r    g    b
    """
    color_map = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required = {"affordance", "r", "g", "b"}

        if reader.fieldnames is None:
            raise ValueError(f"TSV is empty: {path}")

        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "affordance_colors.tsv のヘッダは "
                "affordance, r, g, b が必要です。"
            )

        for row in reader:
            affordance = row["affordance"].strip()

            if not affordance:
                continue

            color_map[affordance] = (
                int(row["r"]),
                int(row["g"]),
                int(row["b"]),
            )

    return color_map


def find_gt_image(gt_dir, stem):
    """
    入力画像と同じstemを持つGT画像を検索する。
    """
    for ext in IMAGE_EXTS:
        candidate = os.path.join(gt_dir, stem + ext)

        if os.path.isfile(candidate):
            return candidate

    return None


def load_affordance_gt(path, size, target_color):
    """
    カラーAnnotation画像から、指定affordance色のみを抽出する。
    """
    gt = Image.open(path).convert("RGB")

    if gt.size != size:
        gt = gt.resize(size, Image.NEAREST)

    gt_np = np.array(gt)
    target_color_np = np.array(target_color, dtype=gt_np.dtype)

    return np.all(
        gt_np == target_color_np,
        axis=-1,
    )


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0

    return intersection / union


def merge_prediction_masks(pred_masks, gt_shape):
    """
    同一affordanceに対して複数のSAM3 maskが返った場合はORで統合する。
    """
    if len(pred_masks) == 0:
        return np.zeros(gt_shape, dtype=bool)

    return np.any(pred_masks, axis=0)


def save_overlay(image_np, pred_mask, color, out_path):
    """
    予測affordance領域を、そのaffordanceのGT色でoverlayする。
    """
    overlay = image_np.copy()
    alpha = 0.5

    for channel in range(3):
        overlay[..., channel] = np.where(
            pred_mask,
            (
                overlay[..., channel] * (1 - alpha)
                + color[channel] * alpha
            ).astype(np.uint8),
            overlay[..., channel],
        )

    Image.fromarray(overlay).save(out_path)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # affordance名 -> RGB色
    color_map = load_affordance_colors(args.affordance_colors)

    if args.affordance not in color_map:
        raise KeyError(
            f"Affordance '{args.affordance}' が "
            f"{args.affordance_colors} に登録されていません。"
        )

    target_color = color_map[args.affordance]

    print(
        f"Affordance: {args.affordance} "
        f"-> GT color: {target_color}"
    )

    # SAM3はこのスクリプト実行中に1回だけロードする
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    image_names = sorted(
        filename
        for filename in os.listdir(args.image_dir)
        if filename.lower().endswith(IMAGE_EXTS)
    )

    rows = []

    for image_name in image_names:
        stem = os.path.splitext(image_name)[0]

        image_path = os.path.join(
            args.image_dir,
            image_name,
        )

        gt_path = find_gt_image(
            args.gt_dir,
            stem,
        )

        if gt_path is None:
            print(f"[skip] GTがありません: {stem}")
            continue

        image = Image.open(image_path).convert("RGB")

        # affordance用語をそのままSAM3のtext promptとして使用
        inference_state = processor.set_image(image)

        output = processor.set_text_prompt(
            state=inference_state,
            prompt=args.affordance,
        )

        masks = output["masks"]
        masks_np = masks.float().cpu().numpy()

        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        pred_masks = masks_np > args.threshold

        # affordanceに対応するGT色だけをbinary mask化
        gt_mask = load_affordance_gt(
            gt_path,
            image.size,
            target_color,
        )

        # 複数予測maskを1枚のaffordance maskへ統合
        pred_affordance_mask = merge_prediction_masks(
            pred_masks,
            gt_mask.shape,
        )

        iou = calculate_iou(
            pred_affordance_mask,
            gt_mask,
        )

        print(
            f"{image_name}: "
            f"IoU={iou:.4f}, "
            f"pred_masks={len(pred_masks)}"
        )

        rows.append(
            {
                "camera_type": args.camera_type,
                "object_name": args.object_name,
                "num_objects": args.num_objects,
                "prompt_id": args.prompt_id,
                "affordance": args.affordance,
                "gt_color_r": target_color[0],
                "gt_color_g": target_color[1],
                "gt_color_b": target_color[2],
                "image": image_name,
                "iou": iou,
                "num_pred_masks": len(pred_masks),
                "gt_pixels": int(gt_mask.sum()),
                "pred_pixels": int(pred_affordance_mask.sum()),
            }
        )

        save_overlay(
            np.array(image),
            pred_affordance_mask,
            target_color,
            os.path.join(
                args.output_dir,
                f"{stem}_overlay.png",
            ),
        )

    mean_iou = (
        float(np.mean([row["iou"] for row in rows]))
        if rows
        else 0.0
    )

    fieldnames = [
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

    with open(
        args.csv_out,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

        writer.writerow(
            {
                "camera_type": args.camera_type,
                "object_name": args.object_name,
                "num_objects": args.num_objects,
                "prompt_id": args.prompt_id,
                "affordance": args.affordance,
                "gt_color_r": target_color[0],
                "gt_color_g": target_color[1],
                "gt_color_b": target_color[2],
                "image": "MEAN",
                "iou": mean_iou,
                "num_pred_masks": "",
                "gt_pixels": "",
                "pred_pixels": "",
            }
        )

    print()
    print(f"評価画像数: {len(rows)}")
    print(f"平均IoU: {mean_iou:.4f}")
    print(f"CSV出力先: {args.csv_out}")


if __name__ == "__main__":
    main()
