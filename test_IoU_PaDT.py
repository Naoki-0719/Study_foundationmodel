import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor

from PaDT import (
    PaDTForConditionalGeneration,
    VisonTextProcessingClass,
    parseVRTintoCompletion,
)
from qwen_vl_utils import process_vision_info


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ------------------------------------------------------------
# GTと対応できなかったPrediction用のfallback色
# matched Predictionには、対応先GTと同じRGB色を使用する。
# ------------------------------------------------------------
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
    (192, 64, 0),
    (64, 192, 0),
    (0, 64, 192),
    (192, 0, 64),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PaDT text-prompted segmentation against GT masks using IoU."
    )

    parser.add_argument("--image_dir", required=True, help="入力画像が入ったディレクトリ")
    parser.add_argument("--gt_dir", required=True, help="対応するGT画像が入ったディレクトリ")

    parser.add_argument(
        "--segmentation_type",
        required=True,
        choices=["semantic", "instance"],
        help="評価形式: semantic または instance",
    )

    # 【PaDT用】TSVには長い命令文を書かず、prompt_typeとtargetのみ持たせる。
    parser.add_argument(
        "--prompt_type",
        required=True,
        choices=["ovd", "rec"],
        help="PaDT prompt形式: ovd または rec",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="検出対象。例: cup, drinking cup, the cup on the left",
    )

    parser.add_argument(
        "--model_path",
        default="PaDT-MLLM/PaDT_Pro_3B",
        help="PaDTモデルのパス",
    )
    parser.add_argument("--device_index", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--max_side",
        type=int,
        default=644,
        help="PaDT入力画像の最大辺。公式demoに合わせて644",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="予測マスクのsigmoid後二値化閾値",
    )

    parser.add_argument(
        "--output_dir",
        default="masks_out_PaDT",
        help="overlay画像の保存先",
    )
    parser.add_argument(
        "--csv_out",
        default="iou_results_PaDT.csv",
        help="画像単位IoUのCSV出力先",
    )
    parser.add_argument(
        "--instance_csv_out",
        default="instance_iou_results_PaDT.csv",
        help="Instance単位IoUのCSV出力先",
    )

    # 実験条件をCSVにも残すための引数
    parser.add_argument("--camera_type", default="", help="カメラ/センサ種別")
    parser.add_argument("--object_name", default="", help="対象物名称")
    parser.add_argument("--num_objects", type=int, default=-1, help="画像内の対象物数")

    return parser.parse_args()


def build_padt_prompt(prompt_type, target):
    """
    【PaDT用】test_demo.pyのPrompt形式に合わせて完全な命令文を生成する。

    ovd: Open Vocabulary Detection。複数同種物体の検出に利用。
    rec: Referring Expression Comprehension。特定物体を文章で指す場合に利用。
    """
    if prompt_type == "ovd":
        return (
            "Please carefully check the image and detect "
            f'the following objects: ["{target}"].'
        )

    if prompt_type == "rec":
        return (
            "Please carefully check the image and detect "
            f'the object this sentence describes: "{target}".'
        )

    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def load_padt_model(model_path, device_index):
    """【PaDT用】test_demo.pyと同じ方法でmodel/processorを準備する。"""
    device = f"cuda:{device_index}"

    model = PaDTForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device_index},
    )

    processor = AutoProcessor.from_pretrained(model_path)
    processor = VisonTextProcessingClass(
        processor,
        model.config.vision_config.spatial_merge_size,
    )
    processor.prepare(model.model.embed_tokens.weight.shape[0])

    model.eval()
    return model, processor, device


def resize_padt_input_images(image_inputs, max_side):
    """【PaDT用】test_demo.pyと同様に最大辺をmax_sideへ合わせる。"""
    resized_images = []

    for image in image_inputs:
        im_w, im_h = image.size
        scale = max_side / max(im_w, im_h)
        new_w = int(im_w * scale)
        new_h = int(im_h * scale)

        resized_images.append(
            image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        )

    return resized_images


def get_padt_eval_size(original_size, max_side):
    """
    【PaDT用】test_demo.pyの可視化と同様に、最大辺をmax_sideへ合わせた後、
    縦横を28の倍数へ丸めたサイズを返す。
    """
    original_w, original_h = original_size
    scale = max_side / max(original_w, original_h)

    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)

    resized_w = max(28, round(scaled_w / 28) * 28)
    resized_h = max(28, round(scaled_h / 28) * 28)

    return resized_w, resized_h


def infer_padt_masks(
    model,
    processor,
    device,
    image_path,
    original_size,
    prompt_type,
    target,
    max_side,
    threshold,
):
    """
    【PaDT用 推論処理】

    test_demo.pyの流れを評価用に整理:
      1. 画像+promptをmessage化
      2. process_vision_info
      3. 最大辺644へリサイズ
      4. model.generate
      5. parseVRTintoCompletion
      6. model.vl_decode
      7. decoded_list['pred_mask'] を取得
      8. valid mask領域を切り出し
      9. sigmoid + threshold
     10. 元画像解像度へ戻して [N,H,W] のbinary maskを返す
    """
    full_prompt = build_padt_prompt(prompt_type, target)

    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": full_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(message)
    image_inputs = resize_padt_input_images(image_inputs, max_side)

    prompt_inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    with torch.inference_mode():
        # test_demo.pyと同様にVisual Reference Token IDをglobal IDへ変換
        prompt_inputs["input_ids"] = processor.assign_to_global_vrt_id(
            prompt_inputs["input_ids"],
            prompt_inputs["image_grid_thw"],
        )

        generate_returned_result = model.generate(
            **prompt_inputs,
            use_cache=True,
            max_new_tokens=1024,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # generation後はlocal VRT IDへ戻す
        prompt_completion_ids = processor.assign_to_local_vrt_id(
            generate_returned_result["sequences"],
            prompt_inputs["image_grid_thw"],
        )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        completions, feats, labels, vrts, vrts_feats = parseVRTintoCompletion(
            processor,
            completion_ids,
            generate_returned_result["hidden_states"],
            torch.Tensor([False]),
        )

        low_res_image_embeds = generate_returned_result.past_image_embeds
        high_res_image_embeds = generate_returned_result.past_high_res_image_embeds
        visual_pe = generate_returned_result.past_visual_pe

        decoded_list = model.vl_decode(
            feats,
            low_res_image_embeds,
            high_res_image_embeds,
            prompt_inputs["image_grid_thw"],
            visual_pe,
        )

        pred_masks = decoded_list["pred_mask"]
        pred_mask_valid_hws = torch.stack(
            [
                decoded_list["pred_mask_valid_hw"][0],
                decoded_list["pred_mask_valid_hw"][1],
            ],
            dim=-1,
        )

    # test_demo.pyの出力サイズへ補間してから元画像サイズへ戻す
    eval_w, eval_h = get_padt_eval_size(original_size, max_side)
    original_w, original_h = original_size

    binary_masks = []

    for mask, mask_hw in zip(pred_masks, pred_mask_valid_hws):
        valid_h = int(mask_hw[0].item()) * 4
        valid_w = int(mask_hw[1].item()) * 4

        mask_logits = mask[None, None, :valid_h, :valid_w]

        mask_prob = torch.nn.functional.interpolate(
            mask_logits,
            size=(eval_h, eval_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].sigmoid()

        mask_binary = (mask_prob > threshold).to(torch.uint8).cpu().numpy()

        # IoUは元画像/GT解像度で計算するため元サイズへ戻す
        mask_pil = Image.fromarray(mask_binary * 255)
        mask_original = mask_pil.resize(
            (original_w, original_h),
            Image.NEAREST,
        )

        binary_masks.append(np.array(mask_original) > 0)

    if len(binary_masks) == 0:
        return (
            np.empty((0, original_h, original_w), dtype=bool),
            full_prompt,
        )

    return np.stack(binary_masks, axis=0), full_prompt


def find_gt_image(gt_dir, stem):
    """入力画像と同じstemを持つGT画像を検索する。"""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_semantic_gt(path, size):
    """
    Semantic GTを読み込む。
    背景=黒、それ以外=foregroundとして扱うため、カラーinstance GTも利用可能。
    """
    gt = Image.open(path).convert("RGB")

    if gt.size != size:
        gt = gt.resize(size, Image.NEAREST)

    gt_np = np.array(gt)
    return np.any(gt_np != 0, axis=-1)


def load_instance_gt(path, size):
    """
    1枚のカラーGT画像を色ごとのinstance maskへ分解する。

    想定:
      背景 = (0,0,0)
      各instance = 異なるRGB色
    """
    gt = Image.open(path).convert("RGB")

    if gt.size != size:
        gt = gt.resize(size, Image.NEAREST)

    gt_np = np.array(gt)
    unique_colors = np.unique(gt_np.reshape(-1, 3), axis=0)

    gt_masks = []
    gt_colors = []

    for color in unique_colors:
        color_tuple = tuple(int(v) for v in color)

        if color_tuple == (0, 0, 0):
            continue

        mask = np.all(gt_np == color, axis=-1)

        if not np.any(mask):
            continue

        gt_masks.append(mask)
        gt_colors.append(color_tuple)

    if len(gt_masks) == 0:
        return (
            np.empty((0, gt_np.shape[0], gt_np.shape[1]), dtype=bool),
            [],
        )

    return np.stack(gt_masks, axis=0), gt_colors


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0

    return intersection / union


def evaluate_semantic(pred_masks, gt_mask):
    """全Prediction maskをORで統合してSemantic IoUを計算する。"""
    if len(pred_masks) == 0:
        pred_semantic = np.zeros_like(gt_mask, dtype=bool)
    else:
        pred_semantic = np.any(pred_masks, axis=0)

    return calculate_iou(pred_semantic, gt_mask)


def evaluate_instance(pred_masks, gt_masks):
    """
    Prediction × GTのIoU行列を作り、Hungarian matchingで
    IoU合計が最大になる1対1対応を求める。

    未対応Prediction/GTはIoU=0として画像単位mean IoUに反映する。
    """
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)

    if num_pred == 0 and num_gt == 0:
        return 1.0, [], np.empty((0, 0), dtype=np.float32)

    if num_pred == 0 or num_gt == 0:
        return 0.0, [], np.zeros((num_pred, num_gt), dtype=np.float32)

    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float32)

    for pred_idx in range(num_pred):
        for gt_idx in range(num_gt):
            iou_matrix[pred_idx, gt_idx] = calculate_iou(
                pred_masks[pred_idx],
                gt_masks[gt_idx],
            )

    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)

    matches = []

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        matches.append(
            {
                "pred_index": int(pred_idx),
                "gt_index": int(gt_idx),
                "iou": float(iou_matrix[pred_idx, gt_idx]),
            }
        )

    total_iou = sum(match["iou"] for match in matches)
    denominator = max(num_pred, num_gt)
    mean_iou = total_iou / denominator if denominator > 0 else 1.0

    return mean_iou, matches, iou_matrix


def blend_mask(overlay, mask, color, alpha=0.5):
    for channel in range(3):
        overlay[..., channel] = np.where(
            mask,
            (
                overlay[..., channel] * (1 - alpha)
                + color[channel] * alpha
            ).astype(np.uint8),
            overlay[..., channel],
        )
    return overlay


def save_semantic_overlay(image_np, pred_masks, out_path):
    overlay = image_np.copy()

    if len(pred_masks) > 0:
        merged = np.any(pred_masks, axis=0)
        overlay = blend_mask(overlay, merged, (255, 0, 0))

    Image.fromarray(overlay).save(out_path)


def get_unused_fallback_color(gt_colors, fallback_index):
    """GT色と完全一致しないfallback色を優先して返す。"""
    gt_color_set = set(gt_colors)

    for offset in range(len(FALLBACK_COLORS)):
        color = FALLBACK_COLORS[
            (fallback_index + offset) % len(FALLBACK_COLORS)
        ]
        if color not in gt_color_set:
            return color, fallback_index + offset + 1

    color = FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
    return color, fallback_index + 1


def save_instance_overlay(
    image_np,
    pred_masks,
    matches,
    gt_colors,
    out_path,
):
    """
    matched Predictionは対応先GTと同じRGB色で表示する。
    unmatched Predictionのみfallback色を使う。
    """
    overlay = image_np.copy()
    pred_colors = {}

    for match in matches:
        pred_colors[match["pred_index"]] = gt_colors[match["gt_index"]]

    fallback_index = 0

    for pred_idx, mask in enumerate(pred_masks):
        if pred_idx in pred_colors:
            color = pred_colors[pred_idx]
        else:
            color, fallback_index = get_unused_fallback_color(
                gt_colors,
                fallback_index,
            )

        overlay = blend_mask(overlay, mask, color)

    Image.fromarray(overlay).save(out_path)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 【PaDT用】1回の実行中はモデルを1度だけ読み込み、image_dir内で再利用する。
    model, processor, device = load_padt_model(
        args.model_path,
        args.device_index,
    )

    image_names = sorted(
        filename
        for filename in os.listdir(args.image_dir)
        if filename.lower().endswith(IMAGE_EXTS)
    )

    image_rows = []
    instance_rows = []

    for image_name in image_names:
        stem = os.path.splitext(image_name)[0]
        image_path = os.path.join(args.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # PaDT推論
        pred_masks, full_prompt = infer_padt_masks(
            model=model,
            processor=processor,
            device=device,
            image_path=image_path,
            original_size=image.size,
            prompt_type=args.prompt_type,
            target=args.target,
            max_side=args.max_side,
            threshold=args.threshold,
        )

        gt_path = find_gt_image(args.gt_dir, stem)

        if gt_path is None:
            print(f"[skip] GTがありません: {stem}")
            continue

        if args.segmentation_type == "semantic":
            gt_mask = load_semantic_gt(gt_path, image.size)
            iou = evaluate_semantic(pred_masks, gt_mask)

            print(
                f"{image_name}: IoU={iou:.4f}, "
                f"pred_instances={len(pred_masks)}"
            )

            image_rows.append(
                {
                    "camera_type": args.camera_type,
                    "object_name": args.object_name,
                    "num_objects": args.num_objects,
                    "prompt_type": args.prompt_type,
                    "target": args.target,
                    "image": image_name,
                    "iou": iou,
                    "num_pred": len(pred_masks),
                    "num_gt": 1,
                }
            )

            save_semantic_overlay(
                np.array(image),
                pred_masks,
                os.path.join(args.output_dir, f"{stem}_overlay.png"),
            )

        else:
            gt_masks, gt_colors = load_instance_gt(gt_path, image.size)

            mean_iou, matches, iou_matrix = evaluate_instance(
                pred_masks,
                gt_masks,
            )

            print(
                f"{image_name}: mean IoU={mean_iou:.4f}, "
                f"pred={len(pred_masks)}, GT={len(gt_masks)}"
            )

            # matched instance
            for match in matches:
                gt_color = gt_colors[match["gt_index"]]

                print(
                    f"    pred#{match['pred_index']} "
                    f"<-> GT#{match['gt_index']} "
                    f"color={gt_color} "
                    f"IoU={match['iou']:.4f}"
                )

                instance_rows.append(
                    {
                        "camera_type": args.camera_type,
                        "object_name": args.object_name,
                        "num_objects": args.num_objects,
                        "prompt_type": args.prompt_type,
                        "target": args.target,
                        "image": image_name,
                        "pred_index": match["pred_index"],
                        "gt_index": match["gt_index"],
                        "gt_color_r": gt_color[0],
                        "gt_color_g": gt_color[1],
                        "gt_color_b": gt_color[2],
                        "matched": 1,
                        "iou": match["iou"],
                    }
                )

            # unmatched Prediction = 過検出
            matched_pred_indices = {match["pred_index"] for match in matches}

            for pred_idx in range(len(pred_masks)):
                if pred_idx in matched_pred_indices:
                    continue

                instance_rows.append(
                    {
                        "camera_type": args.camera_type,
                        "object_name": args.object_name,
                        "num_objects": args.num_objects,
                        "prompt_type": args.prompt_type,
                        "target": args.target,
                        "image": image_name,
                        "pred_index": pred_idx,
                        "gt_index": "",
                        "gt_color_r": "",
                        "gt_color_g": "",
                        "gt_color_b": "",
                        "matched": 0,
                        "iou": 0.0,
                    }
                )

            # unmatched GT = 見逃し
            matched_gt_indices = {match["gt_index"] for match in matches}

            for gt_idx in range(len(gt_masks)):
                if gt_idx in matched_gt_indices:
                    continue

                gt_color = gt_colors[gt_idx]

                instance_rows.append(
                    {
                        "camera_type": args.camera_type,
                        "object_name": args.object_name,
                        "num_objects": args.num_objects,
                        "prompt_type": args.prompt_type,
                        "target": args.target,
                        "image": image_name,
                        "pred_index": "",
                        "gt_index": gt_idx,
                        "gt_color_r": gt_color[0],
                        "gt_color_g": gt_color[1],
                        "gt_color_b": gt_color[2],
                        "matched": 0,
                        "iou": 0.0,
                    }
                )

            image_rows.append(
                {
                    "camera_type": args.camera_type,
                    "object_name": args.object_name,
                    "num_objects": args.num_objects,
                    "prompt_type": args.prompt_type,
                    "target": args.target,
                    "image": image_name,
                    "iou": mean_iou,
                    "num_pred": len(pred_masks),
                    "num_gt": len(gt_masks),
                }
            )

            save_instance_overlay(
                np.array(image),
                pred_masks,
                matches,
                gt_colors,
                os.path.join(args.output_dir, f"{stem}_overlay.png"),
            )

    # 画像単位CSV
    if image_rows:
        dataset_mean_iou = float(np.mean([row["iou"] for row in image_rows]))
    else:
        dataset_mean_iou = 0.0

    image_fieldnames = [
        "camera_type",
        "object_name",
        "num_objects",
        "prompt_type",
        "target",
        "image",
        "iou",
        "num_pred",
        "num_gt",
    ]

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=image_fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)
        writer.writerow(
            {
                "camera_type": args.camera_type,
                "object_name": args.object_name,
                "num_objects": args.num_objects,
                "prompt_type": args.prompt_type,
                "target": args.target,
                "image": "MEAN",
                "iou": dataset_mean_iou,
                "num_pred": "",
                "num_gt": "",
            }
        )

    # Instance単位CSV
    if args.segmentation_type == "instance":
        instance_fieldnames = [
            "camera_type",
            "object_name",
            "num_objects",
            "prompt_type",
            "target",
            "image",
            "pred_index",
            "gt_index",
            "gt_color_r",
            "gt_color_g",
            "gt_color_b",
            "matched",
            "iou",
        ]

        with open(args.instance_csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=instance_fieldnames)
            writer.writeheader()
            writer.writerows(instance_rows)

    print()
    print(f"評価画像数: {len(image_rows)}")
    print(f"平均IoU: {dataset_mean_iou:.4f}")
    print(f"画像単位CSV: {args.csv_out}")

    if args.segmentation_type == "instance":
        print(f"Instance単位CSV: {args.instance_csv_out}")


if __name__ == "__main__":
    main()
