#!/usr/bin/env python3
"""Prepare an auditable 7-class object-detection version of Mines v4.

Does not modify the original dataset. Images containing unresolved rare labels
are quarantined (not included in the clean splits), preventing those objects
from silently becoming unlabelled background.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
NEW_NAMES = [
    "dump_truck", "excavator", "grader", "light_vehicle", "person",
    "safety_helmet", "safety_jacket",
]
# Old class ID -> new class ID. Includes manually reviewed truck variants.
REMAP = {1: 0, 2: 0, 3: 0, 4: 1, 6: 2, 8: 3, 9: 4, 11: 0, 12: 5, 13: 6}
# Preserve original images and annotations outside the clean dataset.
REVIEW_IDS = {0, 5, 7, 10}


def parse_label(line: str, filename: Path, line_number: int):
    values = line.split()
    if not values:
        return None
    try:
        cls = int(values[0])
        coords = [float(v) for v in values[1:]]
    except ValueError as exc:
        raise ValueError(f"{filename}:{line_number}: invalid numeric value") from exc
    if cls < 0 or cls > 13:
        raise ValueError(f"{filename}:{line_number}: class ID outside 0..13: {cls}")
    if not all(math.isfinite(v) for v in coords):
        raise ValueError(f"{filename}:{line_number}: non-finite coordinate")
    if any(v < -1e-5 or v > 1 + 1e-5 for v in coords):
        raise ValueError(f"{filename}:{line_number}: coordinate outside [0, 1]")
    if len(coords) == 4:
        x, y, w, h = coords
        kind = "box"
        if w <= 0 or h <= 0:
            raise ValueError(f"{filename}:{line_number}: non-positive box dimensions")
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        kind = "polygon"
        xs, ys = coords[::2], coords[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        raise ValueError(f"{filename}:{line_number}: expected 4 box coords or 6+ polygon coords")
    # A small amount of floating point drift is tolerated, but never silently
    # clamp substantially invalid annotations.
    if min(x1, y1) < -1e-4 or max(x2, y2) > 1 + 1e-4:
        raise ValueError(f"{filename}:{line_number}: box extends beyond image bounds")
    x1, y1, x2, y2 = [max(0.0, min(1.0, n)) for n in (x1, y1, x2, y2)]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{filename}:{line_number}: degenerate box/polygon")
    box = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return cls, kind, box


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("datasets/mines"))
    ap.add_argument("--dest", type=Path, default=Path("datasets/mines_clean"))
    args = ap.parse_args()
    src, dst = args.source.resolve(), args.dest.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise SystemExit("Source and destination must be separate, non-nested directories")
    if not src.is_dir() or not (src / "data.yaml").is_file():
        raise SystemExit(f"Source dataset or data.yaml missing: {src}")
    if dst.exists():
        raise SystemExit(f"Destination already exists: {dst}. Choose a new path; no files were overwritten.")
    with (src / "data.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    expected = [
        "bull_dozer", "dumb_truck", "dump_truck", "dunp_truck", "excavator",
        "front_end_loader", "grader", "hip_of_soil", "light_vehicle",
        "person", "road_sign", "truck", "z_safety_helmet", "z_safety_jacket",
    ]
    if cfg.get("names") != expected or cfg.get("nc") != len(expected):
        raise SystemExit("Unexpected original class mapping. Inspect data.yaml before remapping.")

    # Validate the complete source first, before creating destination.
    jobs = []
    errors = []
    for split in SPLITS:
        img_dir, label_dir = src / split / "images", src / split / "labels"
        if not img_dir.is_dir() or not label_dir.is_dir():
            raise SystemExit(f"Missing split directories for {split}")
        images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        stems = [p.stem for p in images]
        if len(set(stems)) != len(stems):
            raise SystemExit(f"Ambiguous duplicate image stems in {img_dir}")
        orphan_labels = {p.stem for p in label_dir.glob("*.txt")} - set(stems)
        if orphan_labels:
            raise SystemExit(f"Orphan label files in {split}: {sorted(orphan_labels)[:5]}")
        for image in images:
            label = label_dir / (image.stem + ".txt")
            if not label.is_file():
                errors.append(f"Missing annotation file for {image}")
                continue
            parsed = []
            for lineno, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    parsed.append(parse_label(line, label, lineno))
                except ValueError as exc:
                    errors.append(str(exc))
            jobs.append((split, image, label, parsed))
    if errors:
        raise SystemExit("Dataset validation failed (nothing written):\n" + "\n".join(errors[:20]) + (f"\n... {len(errors)} total errors" if len(errors) > 20 else ""))

    dst.mkdir(parents=True)
    stats = {s: {"images_total": 0, "images_clean": 0, "images_review": 0,
                 "original_box": 0, "original_polygon": 0, "converted_polygon": 0,
                 "instances_clean": 0, "class_counts": Counter()} for s in SPLITS}
    review_rows = []
    for split, image, label, parsed in jobs:
        s = stats[split]
        s["images_total"] += 1
        for cls, kind, _ in parsed:
            s["original_polygon" if kind == "polygon" else "original_box"] += 1
        review = sorted({cls for cls, _, _ in parsed if cls in REVIEW_IDS})
        if review:
            s["images_review"] += 1
            review_rows.append({
                "split": split, "image": image.name, "label": label.name,
                "review_class_ids": ",".join(map(str, review)),
                "review_class_names": ",".join(expected[c] for c in review),
                "all_annotations": len(parsed),
            })
            continue
        out_images = dst / split / "images"
        out_labels = dst / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, out_images / image.name)
        lines = []
        for cls, kind, box in parsed:
            if cls not in REMAP:
                raise RuntimeError(f"Unmapped class {cls} in {label}")
            new_cls = REMAP[cls]
            s["class_counts"][NEW_NAMES[new_cls]] += 1
            s["instances_clean"] += 1
            if kind == "polygon":
                s["converted_polygon"] += 1
            lines.append(f"{new_cls} " + " ".join(f"{v:.8f}" for v in box))
        (out_labels / label.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        s["images_clean"] += 1

    # Absolute dataset root avoids the ambiguity of Roboflow's ../train/images.
    clean_yaml = {
        "path": str(dst), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": len(NEW_NAMES), "names": NEW_NAMES,
    }
    (dst / "data.yaml").write_text(yaml.safe_dump(clean_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with (dst / "review_images.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "label", "review_class_ids", "review_class_names", "all_annotations"])
        writer.writeheader()
        writer.writerows(review_rows)
    report = {
        "source": str(src), "destination": str(dst), "original_class_names": expected,
        "clean_class_names": NEW_NAMES,
        "remap_old_id_to_new_id": {str(k): v for k, v in sorted(REMAP.items())},
        "review_old_class_ids": sorted(REVIEW_IDS),
        "policy": "Exclude entire image if any annotation has an unresolved rare class; never strip such labels and treat the object as background.",
        "splits": {s: {**{k: v for k, v in d.items() if k != "class_counts"}, "class_counts": dict(d["class_counts"])} for s, d in stats.items()},
        "review_images": review_rows,
    }
    (dst / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created: {dst}")
    print(f"{'split':<8} {'source':>8} {'clean':>8} {'review':>8} {'poly converted':>16} {'instances':>12}")
    for split in SPLITS:
        s = stats[split]
        print(f"{split:<8} {s['images_total']:>8} {s['images_clean']:>8} {s['images_review']:>8} {s['converted_polygon']:>16} {s['instances_clean']:>12}")
    print(f"Audit: {dst / 'audit.json'}")
    print(f"Review list: {dst / 'review_images.csv'}")
    print("NOTE: review images remain untouched in the source dataset, not copied into clean splits.")


if __name__ == "__main__":
    main()
