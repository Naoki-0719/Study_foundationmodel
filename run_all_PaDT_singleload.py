import argparse
import csv
import os

import numpy as np
from PIL import Image

# 既存のPaDT評価コードから共通処理を再利用する
from test_IoU_PaDT import (
    IMAGE_EXTS,
    load_padt_model,
    infer_padt_masks,
    find_gt_image,
    load_semantic_gt,
    load_instance_gt,
    evaluate_semantic,
    evaluate_instance,
    save_semantic_overlay,
    save_instance_overlay,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run PaDT IoU evaluation over all dataset conditions "
            "while loading PaDT only once."
        )
    )

    parser.add_argument("--data_root", default="./Data")
    parser.add_argument("--gt_root", default="./GT")
    parser.add_argument("--result_root", default="./results_PaDT_singleload")
    parser.add_argument("--prompt_file", default="./prompts_PaDT.tsv")

    parser.add_argument(
        "--segmentation_type",
        choices=["semantic", "instance"],
        default="instance",
    )

    parser.add_argument(
        "--model_path",
        default="PaDT-MLLM/PaDT_Pro_3B",
    )

    parser.add_argument(
        "--device_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_side",
        type=int,
        default=644,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def load_prompt_table(path):
    """
    prompts_PaDT.tsv を読み込む。

    想定:
        object  prompt_id  prompt_type  target
    """

    prompt_map = {}

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n\r").split("\t")

        expected = ["object", "prompt_id", "prompt_type", "target"]

        if header != expected:
            raise ValueError(
                f"Unexpected TSV header: {header}\n"
                f"Expected: {expected}"
            )

        for line in f:
            line = line.rstrip("\n\r")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 4:
                raise ValueError(
                    f"Invalid TSV row: {line}"
                )

            object_name, prompt_id, prompt_type, target = parts

            prompt_map.setdefault(
                object_name,
                [],
            ).append(
                {
                    "prompt_id": prompt_id,
                    "prompt_type": prompt_type,
                    "target": target,
                }
            )

    return prompt_map


def list_subdirs(path):
    """
    指定ディレクトリ直下のサブディレクトリのみ返す。
    """
    if not os.path.isdir(path):
        return []

    return sorted(
        name
        for name in os.listdir(path)
        if os.path.isdir(
            os.path.join(path, name)
        )
    )


def evaluate_condition(
    model,
    processor,
    device,
    image_dir,
    gt_dir,
    result_dir,
    segmentation_type,
    prompt_type,
    target,
    camera_type,
    object_name,
    num_objects,
    max_side,
    threshold,
):
    """
    1つの実験条件を評価する。

    重要:
        model / processor は外から受け取る。
        ここではモデルを読み込まない。

    これにより、PaDTを1回だけロードしたまま
    全条件を連続評価できる。
    """

    overlay_dir = os.path.join(
        result_dir,
        "overlay",
    )

    os.makedirs(
        overlay_dir,
        exist_ok=True,
    )

    image_rows = []
    instance_rows = []

    image_names = sorted(
        filename
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(IMAGE_EXTS)
    )

    for image_name in image_names:

        stem = os.path.splitext(
            image_name
        )[0]

        image_path = os.path.join(
            image_dir,
            image_name,
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # PaDT推論
        #
        # modelはここでロードしない。
        # 最初に1回ロードした同じmodelを全画像・全条件で再利用する。
        # ----------------------------------------------------
        pred_masks, full_prompt = infer_padt_masks(
            model=model,
            processor=processor,
            device=device,
            image_path=image_path,
            original_size=image.size,
            prompt_type=prompt_type,
            target=target,
            max_side=max_side,
            threshold=threshold,
        )

        gt_path = find_gt_image(
            gt_dir,
            stem,
        )

        if gt_path is None:
            print(
                f"[skip] GT not found: {gt_dir}/{stem}"
            )
            continue

        # ====================================================
        # Semantic segmentation
        # ====================================================
        if segmentation_type == "semantic":

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

            image_rows.append(
                {
                    "camera_type": camera_type,
                    "object_name": object_name,
                    "num_objects": num_objects,
                    "prompt_type": prompt_type,
                    "target": target,
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
                    overlay_dir,
                    f"{stem}_overlay.png",
                ),
            )

        # ====================================================
        # Instance segmentation
        # ====================================================
        else:

            gt_masks, gt_colors = load_instance_gt(
                gt_path,
                image.size,
            )

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

            # matched instance
            for match in matches:

                gt_color = gt_colors[
                    match["gt_index"]
                ]

                print(
                    f"    pred#{match['pred_index']} "
                    f"<-> GT#{match['gt_index']} "
                    f"color={gt_color} "
                    f"IoU={match['iou']:.4f}"
                )

                instance_rows.append(
                    {
                        "camera_type": camera_type,
                        "object_name": object_name,
                        "num_objects": num_objects,
                        "prompt_type": prompt_type,
                        "target": target,
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
            matched_pred_indices = {
                match["pred_index"]
                for match in matches
            }

            for pred_idx in range(
                len(pred_masks)
            ):

                if pred_idx in matched_pred_indices:
                    continue

                instance_rows.append(
                    {
                        "camera_type": camera_type,
                        "object_name": object_name,
                        "num_objects": num_objects,
                        "prompt_type": prompt_type,
                        "target": target,
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
            matched_gt_indices = {
                match["gt_index"]
                for match in matches
            }

            for gt_idx in range(
                len(gt_masks)
            ):

                if gt_idx in matched_gt_indices:
                    continue

                gt_color = gt_colors[
                    gt_idx
                ]

                instance_rows.append(
                    {
                        "camera_type": camera_type,
                        "object_name": object_name,
                        "num_objects": num_objects,
                        "prompt_type": prompt_type,
                        "target": target,
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
                    "camera_type": camera_type,
                    "object_name": object_name,
                    "num_objects": num_objects,
                    "prompt_type": prompt_type,
                    "target": target,
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
                os.path.join(
                    overlay_dir,
                    f"{stem}_overlay.png",
                ),
            )

    # --------------------------------------------------------
    # 画像単位CSV
    # --------------------------------------------------------
    image_csv_path = os.path.join(
        result_dir,
        "iou_results.csv",
    )

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

    if image_rows:
        mean_iou = float(
            np.mean(
                [
                    row["iou"]
                    for row in image_rows
                ]
            )
        )
    else:
        mean_iou = 0.0

    with open(
        image_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=image_fieldnames,
        )

        writer.writeheader()
        writer.writerows(image_rows)

        writer.writerow(
            {
                "camera_type": camera_type,
                "object_name": object_name,
                "num_objects": num_objects,
                "prompt_type": prompt_type,
                "target": target,
                "image": "MEAN",
                "iou": mean_iou,
                "num_pred": "",
                "num_gt": "",
            }
        )

    # --------------------------------------------------------
    # Instance単位CSV
    # --------------------------------------------------------
    if segmentation_type == "instance":

        instance_csv_path = os.path.join(
            result_dir,
            "instance_iou_results.csv",
        )

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

        with open(
            instance_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=instance_fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                instance_rows
            )

    print(
        f"[done] mean IoU={mean_iou:.4f} "
        f"-> {result_dir}"
    )

    return mean_iou


def main():
    args = parse_args()

    if not os.path.isdir(
        args.data_root
    ):
        raise FileNotFoundError(
            f"Data root not found: "
            f"{args.data_root}"
        )

    if not os.path.isdir(
        args.gt_root
    ):
        raise FileNotFoundError(
            f"GT root not found: "
            f"{args.gt_root}"
        )

    if not os.path.isfile(
        args.prompt_file
    ):
        raise FileNotFoundError(
            f"Prompt TSV not found: "
            f"{args.prompt_file}"
        )

    os.makedirs(
        args.result_root,
        exist_ok=True,
    )

    prompt_map = load_prompt_table(
        args.prompt_file
    )

    # ========================================================
    # 【重要な変更点】
    #
    # PaDTのロードはここで1回だけ行う。
    #
    # 以降の
    # Sensor × Object × Number × Prompt × Image
    # 全てで同じmodel / processorを再利用する。
    # ========================================================
    print("Loading PaDT model...")

    model, processor, device = load_padt_model(
        args.model_path,
        args.device_index,
    )

    print("PaDT model loaded.")
    print()

    # ========================================================
    # 全条件をPython内で連続実行
    # ========================================================
    for sensor in list_subdirs(
        args.data_root
    ):

        sensor_dir = os.path.join(
            args.data_root,
            sensor,
        )

        for object_name in list_subdirs(
            sensor_dir
        ):

            object_dir = os.path.join(
                sensor_dir,
                object_name,
            )

            if object_name not in prompt_map:
                print(
                    f"[skip] No prompt: "
                    f"{object_name}"
                )
                continue

            for num_objects in list_subdirs(
                object_dir
            ):

                image_dir = os.path.join(
                    object_dir,
                    num_objects,
                )

                gt_dir = os.path.join(
                    args.gt_root,
                    sensor,
                    object_name,
                    num_objects,
                )

                if not os.path.isdir(
                    gt_dir
                ):
                    print(
                        f"[skip] GT not found: "
                        f"{gt_dir}"
                    )
                    continue

                for prompt_info in prompt_map[
                    object_name
                ]:

                    prompt_id = prompt_info[
                        "prompt_id"
                    ]

                    prompt_type = prompt_info[
                        "prompt_type"
                    ]

                    target = prompt_info[
                        "target"
                    ]

                    result_dir = os.path.join(
                        args.result_root,
                        sensor,
                        object_name,
                        num_objects,
                        prompt_id,
                    )

                    os.makedirs(
                        result_dir,
                        exist_ok=True,
                    )

                    print()
                    print(
                        "============================================================"
                    )
                    print(
                        "Method            : PaDT"
                    )
                    print(
                        f"Sensor            : "
                        f"{sensor}"
                    )
                    print(
                        f"Object            : "
                        f"{object_name}"
                    )
                    print(
                        f"Number of objects : "
                        f"{num_objects}"
                    )
                    print(
                        f"Prompt ID         : "
                        f"{prompt_id}"
                    )
                    print(
                        f"Prompt type       : "
                        f"{prompt_type}"
                    )
                    print(
                        f"Target            : "
                        f"{target}"
                    )
                    print(
                        "============================================================"
                    )

                    evaluate_condition(
                        model=model,
                        processor=processor,
                        device=device,
                        image_dir=image_dir,
                        gt_dir=gt_dir,
                        result_dir=result_dir,
                        segmentation_type=args.segmentation_type,
                        prompt_type=prompt_type,
                        target=target,
                        camera_type=sensor,
                        object_name=object_name,
                        num_objects=num_objects,
                        max_side=args.max_side,
                        threshold=args.threshold,
                    )

    print()
    print(
        "All PaDT evaluations finished."
    )


if __name__ == "__main__":
    main()
