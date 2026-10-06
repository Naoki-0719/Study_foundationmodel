import argparse
import csv
import os

import torch

# CUDA上でbfloat16のautocastを有効化
# SAM3の推論時にメモリ使用量を減らし、高速化する目的
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# 対応する画像拡張子
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# オーバーレイ表示用の色
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
]


def parse_args():
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="Evaluate SAM3 text-prompted segmentation against GT masks using IoU."
    )
    parser.add_argument("--image_dir", required=True, help="入力画像が入ったディレクトリ")
    parser.add_argument("--gt_dir", required=True, help="GTマスクのルートディレクトリ")
    parser.add_argument(
        "--segmentation_type",
        required=True,
        choices=["semantic", "instance"],
        help="評価形式: semantic または instance",
    )
    parser.add_argument("--prompt", required=True, help="SAM3に与えるテキストプロンプト")
    parser.add_argument(
        "--output_dir", default="masks_out", help="予測マスクのオーバーレイ画像保存先"
    )
    parser.add_argument(
        "--csv_out", default="iou_results.csv", help="IoU結果を書き出すCSVファイル"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="予測マスクを二値化する閾値"
    )
    return parser.parse_args()


def find_semantic_gt(gt_dir, stem):
    """
    Semantic segmentation用GTを検索する。

    想定:
        gt_dir/
            semantic/
                image001.png
                image002.png
    """
    semantic_dir = os.path.join(gt_dir, "semantic")

    for ext in IMAGE_EXTS:
        gt_path = os.path.join(semantic_dir, stem + ext)
        if os.path.isfile(gt_path):
            return gt_path

    return None


def find_instance_gt(gt_dir, stem):
    """
    Instance segmentation用GTを検索する。

    想定:
        gt_dir/
            instance/
                image001/
                    000.png
                    001.png
    """
    instance_dir = os.path.join(gt_dir, "instance", stem)

    if not os.path.isdir(instance_dir):
        return []

    return sorted(
        os.path.join(instance_dir, filename)
        for filename in os.listdir(instance_dir)
        if filename.lower().endswith(IMAGE_EXTS)
    )


def load_binary_mask(path, size):
    """GTマスクを読み込み、bool型のbinary maskへ変換する。"""
    mask = Image.open(path).convert("L")

    # 入力画像とサイズが異なる場合はNearest Neighborで合わせる
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)

    return np.array(mask) > 127


def calculate_iou(mask1, mask2):
    """2つのbinary mask間のIoUを計算する。"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # 両方とも空マスクなら完全一致として扱う
    if union == 0:
        return 1.0

    return intersection / union


def evaluate_semantic(pred_masks, gt_mask):
    """
    Semantic segmentation評価。

    SAM3が返した複数のマスクをORで統合し、
    1枚のsemantic maskとしてGTとIoUを計算する。
    """
    if len(pred_masks) == 0:
        pred_semantic = np.zeros_like(gt_mask, dtype=bool)
    else:
        pred_semantic = np.any(pred_masks, axis=0)

    return calculate_iou(pred_semantic, gt_mask)


def evaluate_instance(pred_masks, gt_masks):
    """
    Instance segmentation評価。

    各予測マスクと各GTマスクのIoU行列を作成し、
    IoUが高い組から1対1で対応付ける。
    """
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)

    if num_pred == 0 and num_gt == 0:
        return 1.0, []

    if num_pred == 0 or num_gt == 0:
        return 0.0, []

    # [予測数, GT数] のIoU行列を作成
    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float32)

    for pred_idx in range(num_pred):
        for gt_idx in range(num_gt):
            iou_matrix[pred_idx, gt_idx] = calculate_iou(
                pred_masks[pred_idx], gt_masks[gt_idx]
            )

    matched_ious = []
    used_pred = set()
    used_gt = set()

    # 未使用の組から最大IoUを順番に選択する
    while True:
        best_iou = -1.0
        best_pred = -1
        best_gt = -1

        for pred_idx in range(num_pred):
            if pred_idx in used_pred:
                continue

            for gt_idx in range(num_gt):
                if gt_idx in used_gt:
                    continue

                iou = iou_matrix[pred_idx, gt_idx]

                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_idx
                    best_gt = gt_idx

        if best_pred == -1:
            break

        used_pred.add(best_pred)
        used_gt.add(best_gt)

        matched_ious.append(
            {
                "pred_index": best_pred,
                "gt_index": best_gt,
                "iou": float(best_iou),
            }
        )

        if len(used_pred) == num_pred or len(used_gt) == num_gt:
            break

    # 対応しなかった予測/GTはIoU=0として平均に含める
    denominator = max(num_pred, num_gt)
    total_iou = sum(match["iou"] for match in matched_ious)
    mean_iou = total_iou / denominator

    return mean_iou, matched_ious


def save_overlay(image_np, masks_np, out_path):
    """SAM3の予測マスクを入力画像へ色付きで重ねて保存する。"""
    overlay = image_np.copy()
    alpha = 0.5

    for mask_index, mask in enumerate(masks_np):
        color = COLORS[mask_index % len(COLORS)]

        for channel in range(3):
            overlay[..., channel] = np.where(
                mask,
                (
                    overlay[..., channel] * (1 - alpha)
                    + color[channel] * alpha
                ).astype(np.uint8),
                overlay[..., channel],
            )

    Image.fromarray(overlay).save(out_path)


def main():
    # コマンドライン引数を読み込む
    args = parse_args()

    # 出力先を作成
    os.makedirs(args.output_dir, exist_ok=True)

    # SAM3モデルとProcessorを準備
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # 評価対象となる画像ファイルを取得
    image_names = sorted(
        filename
        for filename in os.listdir(args.image_dir)
        if filename.lower().endswith(IMAGE_EXTS)
    )

    rows = []

    # 各入力画像についてSAM3推論とIoU評価を行う
    for image_name in image_names:
        stem = os.path.splitext(image_name)[0]
        image_path = os.path.join(args.image_dir, image_name)

        # RGB画像として読み込む
        image = Image.open(image_path).convert("RGB")

        # SAM3に画像をセット
        inference_state = processor.set_image(image)

        # テキストプロンプトを与えてセグメンテーション
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=args.prompt,
        )

        # 評価ではSAM3のmaskのみ使用する
        # scoreやbounding boxは使用しない
        masks = output["masks"]

        # GPU Tensor -> CPU NumPy
        masks_np = masks.float().cpu().numpy()

        # [N, 1, H, W] -> [N, H, W]
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        # SAM3の出力マスクを二値化
        pred_masks = masks_np > args.threshold

        # --------------------------------------------------
        # Semantic segmentation評価
        # --------------------------------------------------
        if args.segmentation_type == "semantic":
            gt_path = find_semantic_gt(args.gt_dir, stem)

            if gt_path is None:
                print(f"[skip] Semantic GTがありません: {stem}")
                continue

            gt_mask = load_binary_mask(gt_path, image.size)

            # 全予測インスタンスを統合してsemantic IoUを計算
            iou = evaluate_semantic(pred_masks, gt_mask)

            print(
                f"{image_name}: IoU={iou:.4f}, "
                f"pred_instances={len(pred_masks)}"
            )

            rows.append(
                {
                    "image": image_name,
                    "iou": iou,
                    "num_pred": len(pred_masks),
                    "num_gt": 1,
                }
            )

        # --------------------------------------------------
        # Instance segmentation評価
        # --------------------------------------------------
        else:
            gt_paths = find_instance_gt(args.gt_dir, stem)

            if len(gt_paths) == 0:
                print(f"[skip] Instance GTがありません: {stem}")
                continue

            gt_masks = [
                load_binary_mask(gt_path, image.size)
                for gt_path in gt_paths
            ]

            # 予測とGTを1対1対応付けして平均IoUを計算
            mean_iou, matches = evaluate_instance(pred_masks, gt_masks)

            print(
                f"{image_name}: mean IoU={mean_iou:.4f}, "
                f"pred={len(pred_masks)}, GT={len(gt_masks)}"
            )

            for match in matches:
                print(
                    f"    pred#{match['pred_index']} "
                    f"<-> GT#{match['gt_index']} "
                    f"IoU={match['iou']:.4f}"
                )

            rows.append(
                {
                    "image": image_name,
                    "iou": mean_iou,
                    "num_pred": len(pred_masks),
                    "num_gt": len(gt_masks),
                }
            )

        # SAM3の予測結果を入力画像上に可視化
        image_np = np.array(image)

        save_overlay(
            image_np,
            pred_masks,
            os.path.join(args.output_dir, f"{stem}_overlay.png"),
        )

    # 全評価画像の平均IoU
    if rows:
        mean_iou = np.mean([row["iou"] for row in rows])
    else:
        mean_iou = 0.0

    # CSVへ評価結果を書き出す
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "iou", "num_pred", "num_gt"],
        )

        writer.writeheader()
        writer.writerows(rows)

        writer.writerow(
            {
                "image": "MEAN",
                "iou": mean_iou,
                "num_pred": "",
                "num_gt": "",
            }
        )

    # 最終結果を表示
    print()
    print(f"評価画像数: {len(rows)}")
    print(f"平均IoU: {mean_iou:.4f}")
    print(f"CSV出力先: {args.csv_out}")


if __name__ == "__main__":
    main()
