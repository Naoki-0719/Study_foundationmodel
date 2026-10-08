import argparse
import csv
import os

import torch

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

import numpy as np
from PIL import Image

# 【修正】Hungarian matching を使うために追加
from scipy.optimize import linear_sum_assignment

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# 【修正】未対応の予測マスク用の色を増やした
# 対応済みの予測マスクは、対応先GTと同じ色を使用する。
FALLBACK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
    (255, 0, 128),
    (0, 255, 128),
    (128, 128, 255),
    (255, 128, 128),
    (128, 255, 128),
    (255, 255, 128),
]


def parse_args():
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
        "--output_dir",
        default="masks_out",
        help="予測マスクのオーバーレイ画像保存先",
    )

    parser.add_argument(
        "--csv_out",
        default="iou_results.csv",
        help="画像単位のIoU結果を書き出すCSVファイル",
    )

    # 【修正箇所】
    # Instance segmentation時に、各instanceの対応関係とIoUを
    # 別CSVへ保存するための出力先を追加
    parser.add_argument(
        "--instance_csv_out",
        default="instance_iou_results.csv",
        help="InstanceごとのIoU結果を書き出すCSVファイル",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="予測マスクを二値化する閾値",
    )

    parser.add_argument("--camera_type", default="", help="カメラ/センサ種別")
    parser.add_argument("--object_name", default="", help="対象物の名称")
    parser.add_argument("--num_objects", type=int, default=-1, help="画像内の対象物数")

    return parser.parse_args()


def find_gt_image(gt_dir, stem):
    """
    画像ファイル名に対応するGT画像を検索する。
    """
    for ext in IMAGE_EXTS:
        gt_path = os.path.join(gt_dir, stem + ext)

        if os.path.isfile(gt_path):
            return gt_path

    return None


def load_semantic_gt(path, size):
    """
    Semantic segmentation用GTをbinary maskとして読み込む。
    """
    gt = Image.open(path).convert("L")

    if gt.size != size:
        gt = gt.resize(size, Image.NEAREST)

    return np.array(gt) > 127


# ============================================================
# 【修正箇所】
# Instance segmentation用GTは「1枚のカラー画像」から読み込む。
# 各RGB色を1つのinstance IDとして扱い、色ごとにbinary maskへ分解する。
# また、GTの色そのものも保持してoverlay時に利用する。
# ============================================================
def load_instance_gt(path, size):
    """
    カラーInstance GTを読み込む。

    想定:
        背景 = (0, 0, 0)
        各instance = 固有のRGB色

    Returns
    -------
    masks:
        shape = [N, H, W] のbool配列

    colors:
        各GT instanceに対応するRGB色
    """
    gt = Image.open(path).convert("RGB")

    if gt.size != size:
        gt = gt.resize(size, Image.NEAREST)

    gt_np = np.array(gt)

    # GT画像内のユニークなRGB色を取得
    unique_colors = np.unique(
        gt_np.reshape(-1, 3),
        axis=0,
    )

    masks = []
    colors = []

    for color in unique_colors:
        color_tuple = tuple(int(v) for v in color)

        # 背景色(黒)はinstanceとして扱わない
        if color_tuple == (0, 0, 0):
            continue

        # このRGB色の画素だけをTrueにする
        mask = np.all(
            gt_np == color,
            axis=-1,
        )

        if not np.any(mask):
            continue

        masks.append(mask)
        colors.append(color_tuple)

    if len(masks) == 0:
        return (
            np.empty(
                (0, gt_np.shape[0], gt_np.shape[1]),
                dtype=bool,
            ),
            [],
        )

    return np.stack(masks, axis=0), colors


def calculate_iou(mask1, mask2):
    """
    2つのbinary mask間のIoUを計算する。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0

    return intersection / union


def evaluate_semantic(pred_masks, gt_mask):
    """
    Semantic segmentation評価。

    SAM3が返した複数マスクをORで統合して、
    semantic GTとのIoUを計算する。
    """
    if len(pred_masks) == 0:
        pred_semantic = np.zeros_like(gt_mask, dtype=bool)
    else:
        pred_semantic = np.any(pred_masks, axis=0)

    return calculate_iou(
        pred_semantic,
        gt_mask,
    )


# ============================================================
# 【修正箇所】
# Greedy matchingを廃止し、Hungarian matchingへ変更。
# 全Prediction × 全GTのIoU行列を作成し、
# IoU合計が最大になる1対1対応を求める。
# ============================================================
def evaluate_instance(pred_masks, gt_masks):
    """
    Instance segmentation評価。

    1. PredictionとGTの全組み合わせでIoUを計算
    2. Hungarian matchingで最適な1対1対応を決定
    3. 各instanceのIoUと画像単位mean IoUを返す
    """

    num_pred = len(pred_masks)
    num_gt = len(gt_masks)

    if num_pred == 0 and num_gt == 0:
        return 1.0, [], np.empty((0, 0), dtype=np.float32)

    if num_pred == 0 or num_gt == 0:
        return (
            0.0,
            [],
            np.zeros(
                (num_pred, num_gt),
                dtype=np.float32,
            ),
        )

    # Prediction × GT のIoU行列
    iou_matrix = np.zeros(
        (num_pred, num_gt),
        dtype=np.float32,
    )

    for pred_idx in range(num_pred):
        for gt_idx in range(num_gt):
            iou_matrix[pred_idx, gt_idx] = calculate_iou(
                pred_masks[pred_idx],
                gt_masks[gt_idx],
            )

    # linear_sum_assignmentは最小化問題を解くので、
    # -IoUを渡してIoU合計を最大化する
    pred_indices, gt_indices = linear_sum_assignment(
        -iou_matrix
    )

    matches = []

    for pred_idx, gt_idx in zip(
        pred_indices,
        gt_indices,
    ):
        matches.append(
            {
                "pred_index": int(pred_idx),
                "gt_index": int(gt_idx),
                "iou": float(
                    iou_matrix[pred_idx, gt_idx]
                ),
            }
        )

    # Prediction数とGT数が異なる場合、
    # unmatched分はIoU=0として平均に含める
    denominator = max(
        num_pred,
        num_gt,
    )

    total_iou = sum(
        match["iou"]
        for match in matches
    )

    mean_iou = total_iou / denominator

    return mean_iou, matches, iou_matrix


def save_semantic_overlay(
    image_np,
    pred_masks,
    out_path,
):
    """
    Semantic segmentation用overlay。
    """
    overlay = image_np.copy()
    alpha = 0.5

    if len(pred_masks) == 0:
        Image.fromarray(overlay).save(out_path)
        return

    merged = np.any(
        pred_masks,
        axis=0,
    )

    color = (255, 0, 0)

    for channel in range(3):
        overlay[..., channel] = np.where(
            merged,
            (
                overlay[..., channel] * (1 - alpha)
                + color[channel] * alpha
            ).astype(np.uint8),
            overlay[..., channel],
        )

    Image.fromarray(overlay).save(out_path)


# ============================================================
# 【修正箇所】
# Instance segmentationのoverlayでは、
# Hungarian matchingで対応付けられたPredictionに
# 「対応先GT instanceと同じRGB色」を割り当てる。
#
# そのためPredictionがGTと正確に一致していれば、
# GT画像とPrediction overlayの色対応も一致する。
#
# unmatched PredictionのみFALLBACK_COLORSを使う。
# ============================================================
def save_instance_overlay(
    image_np,
    pred_masks,
    matches,
    gt_colors,
    out_path,
):
    """
    Instance segmentation用overlay。
    """

    overlay = image_np.copy()
    alpha = 0.5

    # Prediction index -> RGB color
    pred_colors = {}

    for match in matches:
        pred_idx = match["pred_index"]
        gt_idx = match["gt_index"]

        pred_colors[pred_idx] = gt_colors[gt_idx]

    fallback_index = 0

    for pred_idx, mask in enumerate(pred_masks):

        if pred_idx in pred_colors:
            # matched Predictionは対応先GTと同じ色
            color = pred_colors[pred_idx]

        else:
            # unmatched Predictionはfallback色
            color = FALLBACK_COLORS[
                fallback_index % len(FALLBACK_COLORS)
            ]
            fallback_index += 1

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

    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # SAM3モデルの準備
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # 評価対象画像一覧
    image_names = sorted(
        filename
        for filename in os.listdir(args.image_dir)
        if filename.lower().endswith(IMAGE_EXTS)
    )

    rows = []

    # 【修正箇所】
    # InstanceごとのIoUを保存するための行データ
    instance_rows = []

    for image_name in image_names:

        stem = os.path.splitext(
            image_name
        )[0]

        image_path = os.path.join(
            args.image_dir,
            image_name,
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # SAM3推論
        inference_state = processor.set_image(
            image
        )

        output = processor.set_text_prompt(
            state=inference_state,
            prompt=args.prompt,
        )

        # maskのみIoU評価に使用
        masks = output["masks"]

        masks_np = (
            masks
            .float()
            .cpu()
            .numpy()
        )

        # [N, 1, H, W] -> [N, H, W]
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        # SAM3 maskを二値化
        pred_masks = (
            masks_np > args.threshold
        )

        # 対応するGT画像を探す
        gt_path = find_gt_image(
            args.gt_dir,
            stem,
        )

        if gt_path is None:
            print(
                f"[skip] GTがありません: {stem}"
            )
            continue

        # ====================================================
        # Semantic segmentation
        # ====================================================
        if args.segmentation_type == "semantic":

            gt_mask = load_semantic_gt(
                gt_path,
                image.size,
            )

            iou = evaluate_semantic(
                pred_masks,
                gt_mask,
            )

            print(
                f"{image_name}: "
                f"IoU={iou:.4f}, "
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

            save_semantic_overlay(
                np.array(image),
                pred_masks,
                os.path.join(
                    args.output_dir,
                    f"{stem}_overlay.png",
                ),
            )

        # ====================================================
        # Instance segmentation
        # ====================================================
        else:

            # 【修正】カラーGT1枚をinstance mask群へ分解
            gt_masks, gt_colors = load_instance_gt(
                gt_path,
                image.size,
            )

            # 【修正】Hungarian matchingでPredictionとGTを対応付け
            mean_iou, matches, iou_matrix = evaluate_instance(
                pred_masks,
                gt_masks,
            )

            print(
                f"{image_name}: "
                f"mean IoU={mean_iou:.4f}, "
                f"pred={len(pred_masks)}, "
                f"GT={len(gt_masks)}"
            )

            # 各instanceの対応関係とIoUを表示
            for match in matches:

                gt_color = gt_colors[
                    match["gt_index"]
                ]

                print(
                    f"    "
                    f"pred#{match['pred_index']} "
                    f"<-> "
                    f"GT#{match['gt_index']} "
                    f"color={gt_color} "
                    f"IoU={match['iou']:.4f}"
                )

                # 【修正箇所】
                # Instanceごとの対応結果をCSV保存用リストへ追加
                instance_rows.append(
                    {
                        "image": image_name,
                        "pred_index": match["pred_index"],
                        "gt_index": match["gt_index"],
                        "gt_color_r": gt_color[0],
                        "gt_color_g": gt_color[1],
                        "gt_color_b": gt_color[2],
                        "iou": match["iou"],
                    }
                )

            # 【修正箇所】
            # Prediction数 > GT数 の場合、未対応PredictionをIoU=0として記録
            matched_pred_indices = {
                match["pred_index"]
                for match in matches
            }

            for pred_idx in range(len(pred_masks)):
                if pred_idx not in matched_pred_indices:
                    instance_rows.append(
                        {
                            "image": image_name,
                            "pred_index": pred_idx,
                            "gt_index": "",
                            "gt_color_r": "",
                            "gt_color_g": "",
                            "gt_color_b": "",
                            "iou": 0.0,
                        }
                    )

            # 【修正箇所】
            # GT数 > Prediction数 の場合、未対応GTをIoU=0として記録
            matched_gt_indices = {
                match["gt_index"]
                for match in matches
            }

            for gt_idx in range(len(gt_masks)):
                if gt_idx not in matched_gt_indices:
                    gt_color = gt_colors[gt_idx]

                    instance_rows.append(
                        {
                            "image": image_name,
                            "pred_index": "",
                            "gt_index": gt_idx,
                            "gt_color_r": gt_color[0],
                            "gt_color_g": gt_color[1],
                            "gt_color_b": gt_color[2],
                            "iou": 0.0,
                        }
                    )

            rows.append(
                {
                    "image": image_name,
                    "iou": mean_iou,
                    "num_pred": len(pred_masks),
                    "num_gt": len(gt_masks),
                }
            )

            # 【修正】対応先GTと同じ色でPredictionをoverlay
            save_instance_overlay(
                np.array(image),
                pred_masks,
                matches,
                gt_colors,
                os.path.join(
                    args.output_dir,
                    f"{stem}_overlay.png",
                ),
            )

    # Dataset全体の平均IoU
    if rows:
        mean_iou = np.mean(
            [
                row["iou"]
                for row in rows
            ]
        )
    else:
        mean_iou = 0.0

    # CSV保存
    with open(
        args.csv_out,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "iou",
                "num_pred",
                "num_gt",
            ],
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

    # ====================================================
    # 【修正箇所】
    # Instance segmentation時は、各instanceのIoUを別CSVへ保存
    # ====================================================
    if args.segmentation_type == "instance":
        with open(
            args.instance_csv_out,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image",
                    "pred_index",
                    "gt_index",
                    "gt_color_r",
                    "gt_color_g",
                    "gt_color_b",
                    "iou",
                ],
            )

            writer.writeheader()
            writer.writerows(instance_rows)

    print()
    print(f"評価画像数: {len(rows)}")
    print(f"平均IoU: {mean_iou:.4f}")
    print(f"画像単位CSV出力先: {args.csv_out}")

    if args.segmentation_type == "instance":
        print(
            f"Instance単位CSV出力先: "
            f"{args.instance_csv_out}"
        )


if __name__ == "__main__":
    main()
