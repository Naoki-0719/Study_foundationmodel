#!/usr/bin/env python3
"""Local-inference (NO cloud LLM API) evaluation of UAD on custom RGB/color-GT affordance dataset.

No retraining required. UAD's checkpoint/DINOv2 weights are obtained separately.
The UAD backbone is initialized once for every Sensor/Object/Number/prompt.
"""
from __future__ import annotations

import argparse
import csv
import functools
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

# Public model/weights downloads are permitted. API-based embeddings are
# rejected in load_uad before the UAD modules are imported.
# Remove LLM credentials from this evaluation process.
for _key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(_key, None)

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_FIELDS = [
    "camera_type", "object_name", "num_objects", "prompt_id", "affordance",
    "gt_color_r", "gt_color_g", "gt_color_b", "image", "iou",
    "num_pred_masks", "gt_pixels", "pred_pixels", "text_query",
]
SUMMARY_FIELDS = [
    "camera_type", "object_name", "num_objects", "prompt_id", "affordance",
    "num_images", "mean_iou", "csv_path",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate UAD affordance masks using SAM3-compatible RGB ground truth."
    )
    p.add_argument("--data_root", default="./DATA", help="Parent of MeasureData and Annotation")
    p.add_argument("--prompt_file", default="./prompts_affordance.tsv")
    p.add_argument("--affordance_colors", default="./affordance_colors.tsv")
    p.add_argument("--output_root", default="./results_affordance_UAD")
    p.add_argument("--uad_root", default="./unsup-affordance", help="Cloned official UAD repository")
    p.add_argument("--config", default="configs/st_emb.yaml", help="Absolute or UAD-root-relative YAML")
    p.add_argument("--checkpoint", default="checkpoints/st_emb.pth", help="Absolute or UAD-root-relative .pth")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold AFTER resizing")
    p.add_argument("--max_size", type=int, default=336, help="Maximum long-side input length BEFORE x2 UAD transform; 0 means no limit")
    p.add_argument("--text_template", default="{affordance}",
                   help='Text embedding query; placeholders {affordance}, {object}, {sensor}, {number}. Default: just action term')
    p.add_argument("--save_scores", action="store_true", help="Save float32 HxW score maps in scores/*.npy")
    p.add_argument("--check_only", action="store_true", help="Validate data/prompt mappings; do not load UAD or run inference")
    p.add_argument("--continue_on_error", action="store_true", help="Log image inference errors and continue; results exclude failed images")
    p.add_argument("--only_sensor", default=None, help="Optional sensor filter")
    p.add_argument("--only_object", default=None, help="Optional object filter")
    p.add_argument("--only_number", default=None, help="Optional Number directory filter")
    p.add_argument("--only_prompt_id", default=None, help="Optional prompt ID filter")
    args = p.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        p.error("--threshold must be between 0 and 1")
    if args.max_size < 0:
        p.error("--max_size must be nonnegative")
    try:
        args.text_template.format(affordance="grasp", object="ObjectA", sensor="Camera", number="1")
    except (KeyError, ValueError) as exc:
        p.error(f"Invalid --text_template: {exc}")
    return args


def load_prompts(path: Path):
    """TSV schema: object, prompt_id, affordance (one row per object/action pair)."""
    by_object = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or not {"object", "prompt_id", "affordance"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid prompt TSV headers in {path}: expected object, prompt_id, affordance")
        for lineno, row in enumerate(reader, start=2):
            obj = (row["object"] or "").strip()
            pid = (row["prompt_id"] or "").strip()
            action = (row["affordance"] or "").strip()
            if not obj and not pid and not action:
                continue
            if not all((obj, pid, action)):
                raise ValueError(f"Incomplete prompt row {lineno} in {path}")
            if any(p == pid for p, _ in by_object[obj]):
                raise ValueError(f"Duplicate prompt_id '{pid}' for object '{obj}' at line {lineno}")
            by_object[obj].append((pid, action))
    return by_object


def load_colors(path: Path):
    """TSV schema: affordance, r, g, b; exact RGB matches define GT regions."""
    colors = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or not {"affordance", "r", "g", "b"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid color TSV headers in {path}: expected affordance, r, g, b")
        for lineno, row in enumerate(reader, start=2):
            action = (row["affordance"] or "").strip()
            if not action:
                continue
            rgb = tuple(int(row[k]) for k in ("r", "g", "b"))
            if any(channel < 0 or channel > 255 for channel in rgb):
                raise ValueError(f"Invalid RGB at {path}:{lineno}")
            if action in colors:
                raise ValueError(f"Duplicate affordance color: {action}")
            colors[action] = rgb
    return colors


def image_files(folder: Path):
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name)


def make_gt_lookup(folder: Path):
    lookup = {}
    for path in image_files(folder):
        if path.stem in lookup:
            raise ValueError(f"Multiple GT images share stem '{path.stem}' in {folder}")
        lookup[path.stem] = path
    return lookup


def read_gt_binary(gt_path: Path, size: tuple[int, int], rgb: tuple[int, int, int]):
    with Image.open(gt_path) as gt_image:
        gt_image = gt_image.convert("RGB")
        if gt_image.size != size:
            gt_image = gt_image.resize(size, resample=Image.Resampling.NEAREST)
        gt = np.asarray(gt_image)
    return np.all(gt == np.asarray(rgb, dtype=np.uint8), axis=2)


def calculate_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    union = np.logical_or(pred, gt).sum(dtype=np.int64)
    return float(inter / union) if union else 1.0  # same convention as supplied SAM3 code


def prepare_uad_image(image: np.ndarray, max_size: int) -> np.ndarray:
    """As in UAD eval_agd.py: preserve aspect ratio and round dimensions UP to /14."""
    h, w = image.shape[:2]
    if max_size and max(h, w) > max_size:
        scale = max_size / max(h, w)
        h, w = max(1, int(h * scale)), max(1, int(w * scale))
    h = max(14, ((h + 13) // 14) * 14)
    w = max(14, ((w + 13) // 14) * 14)
    with Image.fromarray(image, "RGB") as pil:
        return np.asarray(pil.resize((w, h), resample=Image.Resampling.BILINEAR))


def resize_scores(scores: np.ndarray, target_size: tuple[int, int]):
    """Return original-resolution score map (continuous 0..1); target_size=(W,H)."""
    if scores.ndim != 2:
        raise ValueError(f"UAD must return HxW array, got {scores.shape}")
    with Image.fromarray(scores.astype(np.float32), mode="F") as pil:
        if pil.size != target_size:
            pil = pil.resize(target_size, resample=Image.Resampling.BILINEAR)
        result = np.array(pil, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite UAD score map values")
    return np.clip(result, 0.0, 1.0)


def save_overlay(rgb_image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], dst: Path):
    overlay = rgb_image.copy()
    color_arr = np.asarray(color, dtype=np.float32)
    overlay[mask] = np.round(0.5 * rgb_image[mask].astype(np.float32) + 0.5 * color_arr).astype(np.uint8)
    Image.fromarray(overlay, "RGB").save(dst)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def build_jobs(args, prompts, colors):
    data = Path(args.data_root).resolve()
    measure_root, gt_root = data / "MeasureData", data / "Annotation"
    if not measure_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(f"Expected {measure_root} and {gt_root}")
    jobs = []
    for sensor_dir in sorted(measure_root.iterdir()):
        if not sensor_dir.is_dir() or (args.only_sensor and sensor_dir.name != args.only_sensor):
            continue
        for object_dir in sorted(sensor_dir.iterdir()):
            if not object_dir.is_dir() or (args.only_object and object_dir.name != args.only_object):
                continue
            available_prompts = prompts.get(object_dir.name, [])
            if not available_prompts:
                print(f"[skip] No TSV prompts for object: {object_dir.name}")
                continue
            for count_dir in sorted(object_dir.iterdir()):
                if not count_dir.is_dir() or (args.only_number and count_dir.name != args.only_number):
                    continue
                gt_dir = gt_root / sensor_dir.name / object_dir.name / count_dir.name
                if not gt_dir.is_dir():
                    print(f"[skip] No GT directory: {gt_dir}")
                    continue
                paths = image_files(count_dir)
                gts = make_gt_lookup(gt_dir)
                paired = [(p, gts[p.stem]) for p in paths if p.stem in gts]
                if len(paired) != len(paths):
                    print(f"[skip] No matching GT for {len(paths)-len(paired)} / {len(paths)} images in {count_dir}")
                for pid, affordance in available_prompts:
                    if args.only_prompt_id and pid != args.only_prompt_id:
                        continue
                    if affordance not in colors:
                        raise KeyError(f"Affordance '{affordance}' missing from {args.affordance_colors}")
                    if not paired:
                        print(f"[skip] No matching image-GT pairs for {count_dir} ({pid})")
                        continue
                    jobs.append((sensor_dir.name, object_dir.name, count_dir.name,
                                 pid, affordance, colors[affordance], paired))
    return jobs


def load_uad(args):
    """Load official UAD API without patching/duplicating the architecture."""
    root = Path(args.uad_root).resolve()
    src = root / "src"
    if not (src / "inference.py").is_file():
        raise FileNotFoundError(f"Official UAD src/inference.py not found: {src}; clone the repo first")

    def resolve_uad(relative_or_absolute: str):
        p = Path(relative_or_absolute)
        return p if p.is_absolute() else root / p

    config_path = resolve_uad(args.config)
    checkpoint_path = resolve_uad(args.checkpoint)
    for p in (config_path, checkpoint_path):
        if not p.is_file():
            raise FileNotFoundError(f"UAD file not found: {p}")

    # Reject any nonlocal embedding configuration BEFORE importing UAD's modules.
    # UAD's default for a missing text_embedding field is embeddings_oai.
    import yaml
    with config_path.open("r", encoding="utf-8") as _f:
        cfg = yaml.safe_load(_f)
    option = cfg.get("text_embedding") if isinstance(cfg, dict) else None
    if option != "embeddings_st":
        raise RuntimeError(
            f"SECURITY: Refusing text_embedding={option!r}. "
            "Only local SentenceTransformer embeddings_st is permitted. "
            "OpenAI / GPT / external APIs are forbidden."
        )
    if checkpoint_path.name.lower() in {"oai_emb.pth", "eval_agd.pth"}:
        raise RuntimeError("SECURITY: OpenAI-dependent checkpoint is forbidden")

    # UAD utils/img_utils.py unconditionally calls .cuda() while loading DINOv2.
    # Public pretrained weights can be loaded/downloaded by the official UAD implementation.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Official UAD implementation requires CUDA (DINOv2 loader uses .cuda()).")
    sys.path.insert(0, str(src))
    from inference import AffordanceInference
    from utils.vlm_utils import get_text_embedding_options

    print(f"UAD config: {config_path}")
    print(f"UAD checkpoint: {checkpoint_path}")
    print("Text embedding: embeddings_st (LOCAL ONLY)")
    emb_func = functools.lru_cache(maxsize=256)(get_text_embedding_options("embeddings_st"))
    return AffordanceInference(str(config_path), str(checkpoint_path), emb_func)


def evaluate(args, jobs, inference):
    output_root = Path(args.output_root)
    summary_rows = []
    successes = 0
    failures = 0

    for job_index, (sensor, obj, number, pid, affordance, color, pairs) in enumerate(jobs, start=1):
        result_dir = output_root / sensor / obj / number / pid
        overlay_dir, mask_dir = result_dir / "overlay", result_dir / "mask"
        for folder in (overlay_dir, mask_dir):
            folder.mkdir(parents=True, exist_ok=True)
        if args.save_scores:
            (result_dir / "scores").mkdir(parents=True, exist_ok=True)
        text_query = args.text_template.format(
            affordance=affordance, object=obj, sensor=sensor, number=number
        )
        print(f"\n[{job_index}/{len(jobs)}] {sensor}/{obj}/{number}/{pid}: "
              f"{affordance!r} -> prompt {text_query!r}; images={len(pairs)}", flush=True)
        rows = []
        for rgb_path, gt_path in pairs:
            try:
                with Image.open(rgb_path) as pil:
                    rgb_image = np.asarray(pil.convert("RGB"))
                width, height = rgb_image.shape[1], rgb_image.shape[0]
                gt_mask = read_gt_binary(gt_path, (width, height), color)
                resized = prepare_uad_image(rgb_image, args.max_size)
                # NOTE: threshold=None must be used here. Official UAD thresholds BEFORE
                # upsampling, producing interpolated pseudo-binary values otherwise.
                pred_scores = inference.predict(resized, text_query, thresh=None)
                pred_scores = resize_scores(np.asarray(pred_scores), (width, height))
                pred_mask = pred_scores > args.threshold
                iou = calculate_iou(pred_mask, gt_mask)
                row = {
                    "camera_type": sensor, "object_name": obj, "num_objects": number,
                    "prompt_id": pid, "affordance": affordance,
                    "gt_color_r": color[0], "gt_color_g": color[1], "gt_color_b": color[2],
                    "image": rgb_path.name, "iou": iou,
                    "num_pred_masks": 1,  # one UAD score map, unlike multiple SAM3 instances
                    "gt_pixels": int(gt_mask.sum()), "pred_pixels": int(pred_mask.sum()),
                    "text_query": text_query,
                }
                rows.append(row)
                save_overlay(rgb_image, pred_mask, color, overlay_dir / f"{rgb_path.stem}_overlay.png")
                Image.fromarray(pred_mask.astype(np.uint8) * 255, "L").save(mask_dir / f"{rgb_path.stem}_mask.png")
                if args.save_scores:
                    np.save(result_dir / "scores" / f"{rgb_path.stem}_score.npy", pred_scores)
                successes += 1
                print(f"  {rgb_path.name}: IoU={iou:.4f}")
            except Exception as exc:
                failures += 1
                print(f"  [error] {rgb_path}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
                traceback.print_exc()

        mean_iou = float(np.mean([r["iou"] for r in rows])) if rows else 0.0
        csv_path = result_dir / "iou_results.csv"
        atomic_write_csv(csv_path, CSV_FIELDS, rows + [{
            "camera_type": sensor, "object_name": obj, "num_objects": number,
            "prompt_id": pid, "affordance": affordance,
            "gt_color_r": color[0], "gt_color_g": color[1], "gt_color_b": color[2],
            "image": "MEAN", "iou": mean_iou if rows else "",
            "text_query": text_query,
        }])
        summary_rows.append({
            "camera_type": sensor, "object_name": obj, "num_objects": number,
            "prompt_id": pid, "affordance": affordance,
            "num_images": len(rows), "mean_iou": mean_iou if rows else "",
            "csv_path": str(csv_path),
        })
        print(f"  mean IoU={mean_iou:.4f} (n={len(rows)}) -> {csv_path}")
        atomic_write_csv(output_root / "summary.csv", SUMMARY_FIELDS, summary_rows)

    print(f"\nCompleted: {successes} successful image-prompt evaluations; {failures} failures")
    print(f"Summary: {output_root / 'summary.csv'}")
    if failures:
        return 1
    return 0


def main(argv=None):
    args = parse_args(argv)
    prompts = load_prompts(Path(args.prompt_file))
    colors = load_colors(Path(args.affordance_colors))
    jobs = build_jobs(args, prompts, colors)
    print(f"Found {len(jobs)} Sensor/Object/Number/prompt jobs")
    print(f"Total image-prompt evaluations planned: {sum(len(j[-1]) for j in jobs)}")
    if args.check_only:
        for sensor, obj, number, pid, action, rgb, paired in jobs:
            print(f"  {sensor}/{obj}/{number}/{pid} | {action} | color={rgb} | {len(paired)} images")
        return 0
    if not jobs:
        raise RuntimeError("No evaluable image-GT pairs. Check dataset folders and TSV names.")
    inference = load_uad(args)
    return evaluate(args, jobs, inference)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        sys.exit(2)
