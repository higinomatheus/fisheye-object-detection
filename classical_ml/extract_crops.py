"""Extract annotated WoodScape object crops without creating dataset splits.

Reads original RGB images and 2-D box annotations. Output is exploratory and
must not be used to evaluate models before assigning capture-group splits.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from PIL import Image

CLASS_NAMES = {
    0: 'vehicles',
    1: 'person',
    2: 'bicycle',
    3: 'traffic_light',
    4: 'traffic_sign',
}
CAMERA_SUFFIX = re.compile(r'_(?:FV|RV|MVL|MVR)$', re.IGNORECASE)
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def extract(raw: Path, output: Path, class_name: str, max_crops: int | None) -> None:
    if max_crops is not None and max_crops < 1:
        raise ValueError('--max-crops must be >= 1')
    images_dir = raw / 'rgb_images'
    labels_dir = raw / 'box_2d_annotations'
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError('Expected rgb_images/ and box_2d_annotations/ under --raw')
    if not any(labels_dir.glob('*.txt')):
        raise FileNotFoundError(f'No annotation .txt files found in {labels_dir}')

    chosen_class_id = next((i for i, name in CLASS_NAMES.items() if name == class_name), None)
    crop_dir = output / class_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    index_path = output / f'{class_name}_manifest.csv'
    existing = list(crop_dir.glob('*.png'))
    if existing or index_path.exists():
        raise FileExistsError(
            f'Output already exists for {class_name}; choose a new --output directory to avoid stale crops'
        )

    image_lookup = {}
    for file in images_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            if file.stem in image_lookup:
                raise ValueError(f'Duplicate image stem: {file.stem}')
            image_lookup[file.stem] = file

    processed = invalid = missing = 0
    rows = []
    stop = False
    for label_path in sorted(labels_dir.glob('*.txt')):
        image_path = image_lookup.get(label_path.stem)
        if image_path is None:
            missing += 1
            continue
        with Image.open(image_path) as im:
            image = im.convert('RGB')
            width, height = image.size
            with label_path.open('r', encoding='utf-8-sig', newline='') as handle:
                reader = csv.reader(handle)
                for line_num, fields in enumerate(reader, start=1):
                    if len(fields) < 6:
                        invalid += 1
                        continue
                    try:
                        class_id = int(fields[1].strip())
                        xmin, ymin, xmax, ymax = [float(v.strip()) for v in fields[2:6]]
                    except ValueError:
                        invalid += 1
                        continue
                    if class_id not in CLASS_NAMES:
                        invalid += 1
                        continue
                    if chosen_class_id is not None and class_id != chosen_class_id:
                        continue
                    left = max(0, min(width, int(xmin)))
                    top = max(0, min(height, int(ymin)))
                    right = max(0, min(width, int(__import__('math').ceil(xmax))))
                    bottom = max(0, min(height, int(__import__('math').ceil(ymax))))
                    if right <= left or bottom <= top:
                        invalid += 1
                        continue
                    name = f'{label_path.stem}_line{line_num:04d}.png'
                    image.crop((left, top, right, bottom)).save(crop_dir / name)
                    rows.append({
                        'crop': f'{class_name}/{name}',
                        'image_id': label_path.stem,
                        'capture_id': CAMERA_SUFFIX.sub('', label_path.stem),
                        'class_id': class_id,
                        'class_name': CLASS_NAMES[class_id],
                        'xmin': left, 'ymin': top, 'xmax': right, 'ymax': bottom,
                        'source_image': str(image_path),
                    })
                    processed += 1
                    if max_crops is not None and processed >= max_crops:
                        stop = True
                        break
        if stop:
            break

    columns = ['crop', 'image_id', 'capture_id', 'class_id', 'class_name',
               'xmin', 'ymin', 'xmax', 'ymax', 'source_image']
    with index_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Crops generated: {processed}')
    print(f'Invalid annotations skipped: {invalid}')
    print(f'Annotation files without an image: {missing}')
    print(f'Crops: {crop_dir}')
    print(f'Manifest: {index_path}')
    print('NOTE: No train/val/test split has been created. Do not use for final metrics.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--class-name', choices=['all', *CLASS_NAMES.values()], default='person')
    parser.add_argument('--max-crops', type=int, default=50,
                        help='Max crops for local smoke test (default: 50)')
    args = parser.parse_args()
    extract(args.raw, args.output, args.class_name, args.max_crops)


if __name__ == '__main__':
    main()
