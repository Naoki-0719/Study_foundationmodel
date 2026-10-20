"""Shared memory-bank 1/5-shot affordance segmentation and evaluation.

Local-only inference. The repository and checkpoints must be installed locally.
No training/fine-tuning of foundation model is performed.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque  # [追加] 連結成分探索（外部ライブラリ不要）
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# GT palette; everything not equal to one of these colors becomes background.
LABEL_NAMES = ("background", "grasp", "contain", "pick")
LABEL_COLORS = np.array(
    [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)], dtype=np.uint8
)
CSV_COLUMNS = (
    "method", "camera_type", "object_name", "num_objects", "prompt_id",
    "shots", "k", "image", "variant", "affordance", "iou", "gt_pixels", "pred_pixels",  # [変更] 補正前後の比較
)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--support_image_dir", required=True, help="教師側のMeasureData")
    parser.add_argument("--support_gt_dir", required=True, help="教師側のAnnotation")
    parser.add_argument("--image_dir", required=True, help="推論側のMeasureData（元コード準拠）")
    parser.add_argument("--gt_dir", default=None, help="推論側のAnnotation（省略すれば推論画像のみ）")
    parser.add_argument("--shots", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--k", type=int, default=1, help="k-NN の k （1なら最近傍ラベル）")
    parser.add_argument("--max_per_class", type=int, default=2000, help="メモリバンクに残すクラス別特徴数。0は全数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_purity", type=float, default=0.5, help="GTからのパッチラベル確信度の下限")
    parser.add_argument("--chunk_size", type=int, default=1024, help="類似度を計算するqueryパッチ数/回")

    # [追加] 論理的・空間的リファイン。デフォルトOFFで元の結果との比較を可能にする。
    parser.add_argument("--enable_refine", action="store_true", help="保守的な空間制約リファインを有効化")
    parser.add_argument("--refine_anchor_confidence", type=float, default=0.75,
                        help="高確信度アンカーの閾値（クラス投票差、0〜1）")
    parser.add_argument("--refine_min_component", type=int, default=2,
                        help="除去候補にする孤立成分の最大面積（特徴マップ上のセル数）")
    parser.add_argument("--refine_min_anchor", type=int, default=8,
                        help="containアンカーを成立させる最小面積（特徴マップ上のセル数）")
    parser.add_argument("--refine_margin", type=float, default=0.20,
                        help="containアンカーのBBox周囲に取る余裕（特徴マップの長辺に対する割合）")
    parser.add_argument("--output_dir", default="fewshot_affordance_out")
    parser.add_argument("--csv_out", default=None, help="省略時は output_dir/results.csv")
    parser.add_argument("--bank_out", default=None, help="省略時は output_dir/memory_bank.pt")
    parser.add_argument("--camera_type", default="")
    parser.add_argument("--object_name", default="")
    parser.add_argument("--num_objects", type=int, default=-1)
    parser.add_argument("--prompt_id", default="")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")


def validate_arguments(args: argparse.Namespace) -> None:
    if not (0 <= args.min_purity <= 1):
        raise ValueError("--min_purity は 0〜1 で指定してください")
    if args.k < 1 or args.chunk_size < 1 or args.max_per_class < 0:
        raise ValueError("--k, --chunk_size は正整数、--max_per_class は0以上です")
    # [追加] リファインのパラメータを検証
    if not (0 <= args.refine_anchor_confidence <= 1):
        raise ValueError("--refine_anchor_confidence は0〜1で指定してください")
    if args.refine_min_component < 0 or args.refine_min_anchor < 1 or args.refine_margin < 0:
        raise ValueError("refine_min_component>=0、refine_min_anchor>=1、refine_margin>=0が必要です")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAは使用できません。--device cpu を指定してください")
    for name in ("support_image_dir", "support_gt_dir", "image_dir"):
        path = Path(getattr(args, name))
        if not path.is_dir():
            raise FileNotFoundError(f"ディレクトリが見つかりません: {path}")
    if args.gt_dir is not None and not Path(args.gt_dir).is_dir():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {args.gt_dir}")


def get_image_files(root: Path) -> list[Path]:
    """Subfolders (e.g., number-of-items folders) are handled recursively."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def lookup_gt(root: Path, image_path: Path, image_root: Path) -> Optional[Path]:
    """Prefer matching relative directory+stem; fall back to unique stem."""
    rel = image_path.relative_to(image_root)
    local_dir = root / rel.parent
    if local_dir.is_dir():
        matches = sorted(p for p in local_dir.iterdir() if p.is_file() and
                         p.stem == rel.stem and p.suffix.lower() in IMAGE_EXTS)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"GTの候補が複数あります: {rel}")
    global_matches = [p for p in get_image_files(root) if p.stem == rel.stem]
    if len(global_matches) == 1:
        return global_matches[0]
    if len(global_matches) > 1:
        raise ValueError(f"GT名 '{rel.stem}' が複数のフォルダにあります。フォルダ構造を揃えてください")
    return None


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_gt(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """GT image -> class IDs (H,W): 0=other/background, 1=red, 2=blue, 3=green."""
    with Image.open(path) as im:
        gt = im.convert("RGB")
        if gt.size != image_size:
            gt = gt.resize(image_size, resample=Image.Resampling.NEAREST)
        array = np.asarray(gt)
    labels = np.zeros(array.shape[:2], dtype=np.uint8)
    for cls in range(1, len(LABEL_NAMES)):
        labels[np.all(array == LABEL_COLORS[cls], axis=-1)] = cls
    return labels


def gt_to_feature_labels(gt: np.ndarray, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign to each feature location the most frequent GT class in its pixel cell."""
    gt_tensor = torch.from_numpy(gt.astype(np.int64))
    onehot = F.one_hot(gt_tensor, num_classes=len(LABEL_NAMES)).permute(2, 0, 1).float().unsqueeze(0)
    freq = F.adaptive_avg_pool2d(onehot, (h, w))[0]
    purity, labels = freq.max(dim=0)
    return labels, purity


def roi_to_grid(roi: np.ndarray, h: int, w: int) -> torch.Tensor:
    # Area pooling makes sure that fine/thin ROI is not lost at the feature scale.
    array = torch.from_numpy(roi.astype(np.float32))[None, None]
    pooled = F.adaptive_max_pool2d(array, (h, w))[0, 0]
    return pooled > 0


def flatten_features(fmap: torch.Tensor) -> torch.Tensor:
    """Accept CHW map and return L2-normalized flattened matrix (H*W,C) on CPU."""
    if fmap.ndim != 3:
        raise ValueError(f"想定外の特徴マップサイズ: {tuple(fmap.shape)} (C,H,Wを期待)")
    features = fmap.float().permute(1, 2, 0).reshape(-1, fmap.shape[0])
    return F.normalize(features, dim=1).detach().cpu().contiguous()


def build_memory_bank(args, extractor, support_paths: list[Path], support_root: Path, support_gt_root: Path):
    features_by_class = {i: [] for i in range(len(LABEL_NAMES))}
    for i, image_path in enumerate(support_paths, 1):
        gt_path = lookup_gt(support_gt_root, image_path, support_root)
        if gt_path is None:
            raise FileNotFoundError(f"教師GTがありません: {image_path}")
        image = open_rgb(image_path)
        fmap, roi = extractor.extract(image)
        _, h, w = fmap.shape
        labels, purity = gt_to_feature_labels(load_gt(gt_path, image.size), h, w)
        inside = roi_to_grid(roi, h, w).reshape(-1)
        keep = inside & (purity.reshape(-1) >= args.min_purity)
        f = flatten_features(fmap)
        flat_labels = labels.reshape(-1)
        counts = {}
        for c in range(len(LABEL_NAMES)):
            selection = keep & (flat_labels == c)
            counts[LABEL_NAMES[c]] = int(selection.sum())
            if selection.any():
                features_by_class[c].append(f[selection])
        print(f"[support {i}/{len(support_paths)}] {image_path.relative_to(support_root)}: {counts}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    bank_feats, bank_labels, counts = [], [], {}
    for c in range(len(LABEL_NAMES)):
        if not features_by_class[c]:
            counts[LABEL_NAMES[c]] = 0
            print(f"[warning] {LABEL_NAMES[c]} のメモリ特徴がありません", flush=True)
            continue
        feats = torch.cat(features_by_class[c], dim=0)
        if args.max_per_class > 0 and len(feats) > args.max_per_class:
            ids = torch.randperm(len(feats), generator=gen)[:args.max_per_class]
            feats = feats[ids]
        bank_feats.append(feats)
        bank_labels.append(torch.full((len(feats),), c, dtype=torch.long))
        counts[LABEL_NAMES[c]] = len(feats)
    if not bank_feats:
        raise ValueError("メモリバンクが空です。GT色、ROI、min_purityを確認してください")
    if sum(counts.values()) < args.k:
        raise ValueError(f"メモリ数 {sum(counts.values())} < k={args.k}")
    print(f"[bank] {counts}", flush=True)
    return torch.cat(bank_feats), torch.cat(bank_labels), counts


# [追加] ラベルと「k近傍投票差」による確信度を同時に計算する。
# 重要：これは正解確率ではない。k=1だと投票差は常に1になる。
def classify_knn_with_confidence(
    query_feats: torch.Tensor,
    bank_feats: torch.Tensor,
    bank_labels: torch.Tensor,
    k: int,
    device: str,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine k-NN: return (pred_labels, top1_minus_top2_vote_margin)."""
    b = bank_feats.to(device)
    y = bank_labels.to(device)
    predictions, confidences = [], []

    for chunk in query_feats.split(chunk_size, dim=0):
        sims = chunk.to(device) @ b.T
        values, indices = sims.topk(k, dim=1)
        nearest_labels = y[indices]

        # [修正] autocast(bfloat16) と votes(float32) のdtype不一致を防止。
        weights = ((values.float() + 1.0) / 2.0).clamp(min=1e-6)
        votes = torch.zeros(
            (chunk.shape[0], len(LABEL_NAMES)), device=device, dtype=torch.float32
        )
        votes.scatter_add_(1, nearest_labels, weights)
        scores = votes / votes.sum(dim=1, keepdim=True).clamp_min(1e-8)

        pred = scores.argmax(dim=1)
        top2 = scores.topk(2, dim=1).values
        confidence = (top2[:, 0] - top2[:, 1]).clamp(0, 1)
        predictions.append(pred.cpu())
        confidences.append(confidence.cpu())

    return torch.cat(predictions), torch.cat(confidences)


def classify_knn(query_feats: torch.Tensor, bank_feats: torch.Tensor, bank_labels: torch.Tensor,
                 k: int, device: str, chunk_size: int) -> torch.Tensor:
    """[変更] 従来のラベルだけを返すAPIも維持する（既存コード互換）。"""
    pred, _ = classify_knn_with_confidence(
        query_feats, bank_feats, bank_labels, k, device, chunk_size
    )
    return pred


# [追加] 8近傍で連結成分を求める。scipy / opencv 等の追加依存なし。
def _connected_components_8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        coords = []
        while queue:
            cy, cx = queue.popleft()
            coords.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w and mask[ny, nx]
                            and not visited[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        components.append(np.asarray(coords, dtype=np.int32))
    return components


def refine_by_logical_constraints(
    pred_grid: np.ndarray,
    confidence_grid: np.ndarray,
    roi_grid: np.ndarray | None = None,
    anchor_confidence: float = 0.75,
    min_component: int = 2,
    min_anchor: int = 8,
    margin: float = 0.20,
) -> tuple[np.ndarray, dict[str, int]]:
    """[追加] 高確信度アンカーに基づく、控えめな論理・空間リファイン。

    特徴マップ解像度 (Hf,Wf) で実行する。改変は foreground -> background のみ。
    規則1: 高確信度画素を含まない、ごく小さな孤立した予測成分を除去。
    規則2: 十分な高確信度containアンカーが得られた場合のみ、そこから
           大きく離れた、低確信度のforeground連結成分を除去。

    graspは箱の縁の一部分なので、graspによる閉領域を仮定しない。
    pickは複数の独立領域を許容し、複数のpick領域を一つに連結しない。
    ROIなしでは厳密な箱の内外判定はできないため、これはヒューリスティック。
    """
    pred = np.asarray(pred_grid)
    conf = np.asarray(confidence_grid, dtype=np.float32)
    if pred.ndim != 2 or conf.shape != pred.shape:
        raise ValueError("pred_gridとconfidence_gridの形状が一致する2次元配列である必要があります")
    h, w = pred.shape
    roi = np.ones((h, w), dtype=bool) if roi_grid is None else np.asarray(roi_grid, dtype=bool)
    if roi.shape != pred.shape:
        raise ValueError("roi_gridの形状がpred_gridと一致しません")
    result = pred.copy()
    result[~roi] = 0
    counts = {"tiny_removed": 0, "far_removed": 0, "removed_cells": 0,
              "contain_anchor_available": 0}

    # [追加] 高確信度containの最大連結成分を「箱内側の暫定アンカー」とする。
    # 信頼できるアンカーがなければ、箱の内外に関する除去はしない。
    contain_id = LABEL_NAMES.index("contain")
    anchor_mask = (result == contain_id) & (conf >= anchor_confidence) & roi
    anchor_parts = _connected_components_8(anchor_mask)
    anchor_bbox = None
    if anchor_parts:
        largest = max(anchor_parts, key=len)
        if len(largest) >= min_anchor:
            pad = max(2, int(np.ceil(margin * max(h, w))))
            anchor_bbox = (
                max(0, int(largest[:, 0].min()) - pad),
                min(h - 1, int(largest[:, 0].max()) + pad),
                max(0, int(largest[:, 1].min()) - pad),
                min(w - 1, int(largest[:, 1].max()) + pad),
            )
            counts["contain_anchor_available"] = 1

    # [追加] クラスごとに独立して連結成分を調べる。
    # 高確信度セルを1つでも含む成分は、誤除去を避けるため必ず残す。
    for class_id in range(1, len(LABEL_NAMES)):
        for component in _connected_components_8((result == class_id) & roi):
            ys, xs = component[:, 0], component[:, 1]
            if float(conf[ys, xs].max()) >= anchor_confidence:
                continue

            too_small = len(component) <= min_component
            outside = False
            if anchor_bbox is not None:
                y0, y1, x0, x1 = anchor_bbox
                # 成分全体が、アンカー周辺の大きめの矩形領域から外れている場合のみ。
                outside = bool((ys.max() < y0) or (ys.min() > y1) or
                               (xs.max() < x0) or (xs.min() > x1))

            if too_small or outside:
                result[ys, xs] = 0
                counts["removed_cells"] += len(component)
                if too_small:
                    counts["tiny_removed"] += 1
                else:
                    counts["far_removed"] += 1
    return result, counts


# [追加] 確信度マップをPNG(目視用)とNPY(元のfloat値)で保存。
def save_confidence_map(args, confidence: np.ndarray, rel: Path) -> None:
    out = Path(args.output_dir) / rel.parent
    out.mkdir(parents=True, exist_ok=True)
    values = np.clip(confidence.astype(np.float32), 0.0, 1.0)
    Image.fromarray((values * 255).astype(np.uint8), mode="L").save(
        out / f"{rel.stem}_confidence.png"
    )
    np.save(out / f"{rel.stem}_confidence.npy", values)


def calculate_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def save_visualizations(args, img: Image.Image, predicted: np.ndarray, rel: Path, suffix: str = ""):  # [変更] refined版を別名保存
    output = Path(args.output_dir) / rel.parent
    output.mkdir(parents=True, exist_ok=True)
    stem = rel.stem + suffix
    rgb = LABEL_COLORS[predicted]
    Image.fromarray(rgb, mode="RGB").save(output / f"{stem}_prediction.png")
    src = np.asarray(img).astype(np.float32)
    overlay = src.copy()
    fg = predicted > 0
    overlay[fg] = src[fg] * 0.5 + rgb[fg].astype(np.float32) * 0.5
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(output / f"{stem}_overlay.png")
    for class_id, name in enumerate(LABEL_NAMES[1:], 1):
        binary = ((predicted == class_id).astype(np.uint8) * 255)
        Image.fromarray(binary, mode="L").save(output / f"{stem}_{name}_mask.png")


def run_experiment(args: argparse.Namespace, extractor, method: str) -> None:
    validate_arguments(args)
    support_root, query_root = Path(args.support_image_dir), Path(args.image_dir)
    support_gt_root = Path(args.support_gt_dir)
    query_gt_root = Path(args.gt_dir) if args.gt_dir else None
    support_candidates = get_image_files(support_root)
    valid_support = [p for p in support_candidates if lookup_gt(support_gt_root, p, support_root) is not None]
    if len(valid_support) < args.shots:
        raise ValueError(f"教師GT付き画像が {len(valid_support)} 枚しかありません。shots={args.shots}")
    support_paths = valid_support[:args.shots]  # reproducible, sorted
    support_abs = {p.resolve() for p in support_paths}
    query_paths = [p for p in get_image_files(query_root) if p.resolve() not in support_abs]
    if not query_paths:
        raise ValueError("推論画像がありません（教師画像は重複防止のため除外しました）")
    if args.gt_dir:
        # GT may be in both dirs, but the support samples themselves must not be evaluated.
        query_gt_resolved = {lookup_gt(query_gt_root, p, query_root).resolve()
                             for p in query_paths if lookup_gt(query_gt_root, p, query_root) is not None}
        train_gt_resolved = {lookup_gt(support_gt_root, p, support_root).resolve() for p in support_paths}
        # If images differ but their GT path matches, this is suspicious; prevent leakage.
        if query_gt_resolved & train_gt_resolved:
            raise ValueError("教師と推論で同じGTファイルを参照しています。分割を確認してください")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    bank_feats, bank_labels, bank_counts = build_memory_bank(
        args, extractor, support_paths, support_root, support_gt_root
    )
    bank_path = Path(args.bank_out) if args.bank_out else outdir / "memory_bank.pt"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": bank_feats, "labels": bank_labels,
        "label_names": LABEL_NAMES, "label_colors": LABEL_COLORS.tolist(),
        "method": method, "shots": args.shots, "selected_support_files": [str(p) for p in support_paths],
        "memory_counts": bank_counts,
    }, bank_path)
    print(f"[saved] memory bank: {bank_path}", flush=True)

    rows = []
    for index, path in enumerate(query_paths, 1):
        rel = path.relative_to(query_root)
        image = open_rgb(path)
        fmap, roi = extractor.extract(image)
        _, hf, wf = fmap.shape
        flat = flatten_features(fmap)
        # [変更] k-NNのラベルだけでなく、確信度（上位2クラスの投票差）も取得。
        pred_flat, conf_flat = classify_knn_with_confidence(
            flat, bank_feats, bank_labels, args.k, args.device, args.chunk_size
        )
        pred_grid = pred_flat.reshape(hf, wf).to(torch.uint8)
        conf_grid = conf_flat.reshape(hf, wf).float()

        # [追加] 精度を比較できるようリファイン前のgridを保存したまま処理。
        refined_grid = None
        if args.enable_refine:
            if args.k == 1:
                # 投票差は全画素1になるため、アンカー判定に使わない。
                print("[warning] k=1では投票差確信度が常に1です。リファインをスキップします。"
                      " --k 5 等を指定してください", flush=True)
            else:
                roi_grid = roi_to_grid(roi, hf, wf).numpy()
                refined_np, stats = refine_by_logical_constraints(
                    pred_grid.numpy(), conf_grid.numpy(), roi_grid,
                    anchor_confidence=args.refine_anchor_confidence,
                    min_component=args.refine_min_component,
                    min_anchor=args.refine_min_anchor,
                    margin=args.refine_margin,
                )
                refined_grid = torch.from_numpy(refined_np).to(torch.uint8)
                print(f"[refine] {rel}: {stats}", flush=True)

        # [変更] ラベルはnearest、確信度はbilinearで元画像の解像度に戻す。
        pred = F.interpolate(
            pred_grid[None, None].float(), size=(image.height, image.width), mode="nearest"
        )[0, 0].to(torch.uint8).numpy()
        confidence = F.interpolate(
            conf_grid[None, None], size=(image.height, image.width),
            mode="bilinear", align_corners=False
        )[0, 0].numpy()

        pred[~roi] = 0
        confidence[~roi] = 0.0
        save_visualizations(args, image, pred, rel)
        save_confidence_map(args, confidence, rel)  # [追加] confidence.png と .npy

        refined = None
        if refined_grid is not None:
            refined = F.interpolate(
                refined_grid[None, None].float(), size=(image.height, image.width),
                mode="nearest"
            )[0, 0].to(torch.uint8).numpy()
            refined[~roi] = 0
            save_visualizations(args, image, refined, rel, suffix="_refined")  # [追加]

        gt_path = lookup_gt(query_gt_root, path, query_root) if query_gt_root else None
        if gt_path is None:
            print(f"[query {index}/{len(query_paths)}] {rel} (GTなし: 予測のみ)", flush=True)
            continue
        gt = load_gt(gt_path, image.size)

        # [変更] CSVでraw/refinedを別行として記録し、IoU改善を直接比較可能に。
        for variant, variant_pred in (("raw", pred), ("refined", refined)):
            if variant_pred is None:
                continue
            scores = {}
            for class_id, name in enumerate(LABEL_NAMES[1:], start=1):
                mask = variant_pred == class_id
                gt_mask = gt == class_id
                score = calculate_iou(mask, gt_mask)
                scores[name] = score
                rows.append({
                    "method": method, "camera_type": args.camera_type,
                    "object_name": args.object_name, "num_objects": args.num_objects,
                    "prompt_id": args.prompt_id, "shots": args.shots, "k": args.k,
                    "image": str(rel), "variant": variant,
                    "affordance": name, "iou": score,
                    "gt_pixels": int(gt_mask.sum()), "pred_pixels": int(mask.sum()),
                })
            print(f"[query {index}/{len(query_paths)}] {rel} [{variant}]: " +
                  ", ".join(f"{n}={v:.4f}" for n, v in scores.items()), flush=True)

    if rows:
        # [変更] raw / refined それぞれに対して平均IoUを集計する。
        variants = [v for v in ("raw", "refined") if any(r["variant"] == v for r in rows)]
        for variant in variants:
            means = []
            for name in LABEL_NAMES[1:]:
                vals = [r["iou"] for r in rows if r["affordance"] == name
                        and r["variant"] == variant and r["image"] != "MEAN"]
                mean = float(np.mean(vals))
                means.append(mean)
                print(f"[MEAN] {variant}/{name}: {mean:.4f}", flush=True)
                rows.append({"method": method, "camera_type": args.camera_type,
                             "object_name": args.object_name, "num_objects": args.num_objects,
                             "prompt_id": args.prompt_id, "shots": args.shots, "k": args.k,
                             "image": "MEAN", "variant": variant, "affordance": name,
                             "iou": mean, "gt_pixels": "", "pred_pixels": ""})
            all_mean = float(np.mean(means))
            rows.append({"method": method, "camera_type": args.camera_type,
                         "object_name": args.object_name, "num_objects": args.num_objects,
                         "prompt_id": args.prompt_id, "shots": args.shots, "k": args.k,
                         "image": "MEAN_ALL", "variant": variant, "affordance": "ALL",
                         "iou": all_mean, "gt_pixels": "", "pred_pixels": ""})
            print(f"[MEAN_ALL] {variant}: IoU={all_mean:.4f}", flush=True)
    else:
        print("[info] 評価用GTがないのでIoUは算出しませんでした", flush=True)

    csv_path = Path(args.csv_out) if args.csv_out else outdir / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] results: {csv_path}", flush=True)
