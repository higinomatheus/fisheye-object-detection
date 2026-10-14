#!/usr/bin/env python3
"""Operational bounding-box evaluation on a YOLO-format train/val/test split.

One-to-one matching is performed *within each class*, by descending prediction
confidence, at a separately configured matching-IoU threshold. A wrong-class
prediction therefore counts as a false positive for its predicted class and a
false negative for the ground-truth class unless another prediction matches it.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "woodscape"
CLASSES = ["vehicles", "person", "bicycle", "traffic_light", "traffic_sign"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_labels(path, width, height):
    objects = []
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label {path}:{lineno}")
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        if not 0 <= cls < len(CLASSES):
            raise ValueError(f"Unexpected class {cls} at {path}:{lineno}")
        objects.append((cls, ((xc - w/2)*width, (yc - h/2)*height,
                              (xc + w/2)*width, (yc + h/2)*height)))
    return objects


def match_image(ground_truth, predictions, threshold):
    """Return counts by class for a single image."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    used = set()
    for cls, box, conf in sorted(predictions, key=lambda p: -p[2]):
        best_iou, best_idx = threshold, None
        for i, (gt_cls, gt_box) in enumerate(ground_truth):
            if i in used or gt_cls != cls:
                continue
            overlap = box_iou(box, gt_box)
            if overlap >= best_iou:
                best_iou, best_idx = overlap, i
        if best_idx is None:
            counts[cls]["fp"] += 1
        else:
            used.add(best_idx)
            counts[cls]["tp"] += 1
    for i, (gt_cls, _) in enumerate(ground_truth):
        if i not in used:
            counts[gt_cls]["fn"] += 1
    return counts


def calc_metrics(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def main():
    parser = argparse.ArgumentParser(description="Operational evaluation: confidence + class-specific IoU matching")
    parser.add_argument("--model-type", choices=["yolo", "rtdetr"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Dataset split (default: test; use val for threshold selection)")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--match-iou", type=float, default=0.50, help="IoU threshold for GT-prediction matching")
    parser.add_argument("--nms-iou", type=float, default=0.70, help="IoU parameter passed to model.predict (not matching IoU)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "operational")
    args = parser.parse_args()

    if not 0 <= args.conf <= 1 or not 0 < args.match_iou <= 1 or not 0 < args.nms_iou <= 1:
        parser.error("conf must be 0..1; IoU thresholds must be >0 and <=1")
    if args.batch < 1 or args.imgsz < 1:
        parser.error("batch and imgsz must be positive")
    if not args.weights.is_file():
        parser.error(f"Checkpoint missing: {args.weights}")
    img_dir = args.data_dir / "images" / args.split
    label_dir = args.data_dir / "labels" / args.split
    if not img_dir.is_dir() or not label_dir.is_dir():
        parser.error(f"Missing split directories: {img_dir} or {label_dir}")
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        parser.error(f"No images found in {img_dir}")
    missing = [p.name for p in images if not (label_dir / f"{p.stem}.txt").is_file()]
    if missing:
        parser.error(f"Missing labels for {len(missing)} images (first: {missing[0]})")
    output = args.output_root / args.name
    if output.exists():
        parser.error(f"Output already exists, choose a new --name: {output}")
    output.mkdir(parents=True)
    print(f"Split: {args.split} ({len(images)} images); confidence: {args.conf}; "
          f"matching IoU: {args.match_iou}; NMS IoU: {args.nms_iou}")

    model = RTDETR(str(args.weights)) if args.model_type == "rtdetr" else YOLO(str(args.weights))
    total = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_image = []
    started = time.perf_counter()
    for start in range(0, len(images), args.batch):
        batch = images[start:start + args.batch]
        results = model.predict(
            source=[str(p) for p in batch], conf=args.conf, iou=args.nms_iou,
            imgsz=args.imgsz, device=args.device, verbose=False, save=False,
            batch=len(batch),
        )
        if len(results) != len(batch):
            raise RuntimeError("Prediction batch size does not match input count")
        for path, result in zip(batch, results):
            with Image.open(path) as im:
                width, height = im.size
            gt = read_labels(label_dir / f"{path.stem}.txt", width, height)
            predictions = [
                (int(cls), tuple(float(v) for v in box), float(conf))
                for box, cls, conf in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                )
            ]
            counts = match_image(gt, predictions, args.match_iou)
            row = {"image": path.name, "gt": len(gt), "predictions": len(predictions),
                   "tp": sum(v["tp"] for v in counts.values()),
                   "fp": sum(v["fp"] for v in counts.values()),
                   "fn": sum(v["fn"] for v in counts.values())}
            per_image.append(row)
            for cls, values in counts.items():
                for key, value in values.items():
                    total[cls][key] += value
        print(f"Processed {min(start + len(batch), len(images))}/{len(images)} images", end="\r", flush=True)
    seconds = time.perf_counter() - started
    print()

    rows = []
    for cls, name in enumerate(CLASSES):
        rows.append({"class_id": cls, "class_name": name, **calc_metrics(total[cls])})
    pooled = {k: sum(total[c][k] for c in range(len(CLASSES))) for k in ("tp", "fp", "fn")}
    micro = calc_metrics(pooled)
    macro = {k: sum(row[k] for row in rows) / len(rows) for k in ("precision", "recall", "f1")}
    config = dict(model_type=args.model_type, weights=str(args.weights), data_dir=str(args.data_dir),
                  split=args.split, image_count=len(images), conf=args.conf, match_iou=args.match_iou,
                  nms_iou=args.nms_iou, imgsz=args.imgsz, batch=args.batch, device=args.device,
                  elapsed_wall_seconds=seconds, wall_ms_per_image=seconds * 1000 / len(images),
                  note="Wall time includes image loading, preprocessing, inference and matching; not GPU-only inference speed.",
                  caveat="Greedy one-to-one class-aware matching. Not Ultralytics mAP; not a classification accuracy metric.")
    with (output / "per_class.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    (output / "summary.json").write_text(json.dumps({"config": config, "micro": micro, "macro": macro,
                                                       "classes": rows}, indent=2, ensure_ascii=False))
    print("\nClass           TP    FP    FN   Precision  Recall    F1")
    for row in rows:
        print(f"{row['class_name']:<14} {row['tp']:5} {row['fp']:5} {row['fn']:5}  "
              f"{row['precision']:.4f}     {row['recall']:.4f}  {row['f1']:.4f}")
    print(f"\nMicro: P={micro['precision']:.4f}, R={micro['recall']:.4f}, F1={micro['f1']:.4f}")
    print(f"Wall time: {seconds:.1f}s / {len(images)} images (includes I/O)")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()