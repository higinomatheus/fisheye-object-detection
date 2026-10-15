
from pathlib import Path
from collections import defaultdict
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import yaml


DATASET_DIR = Path("datasets/mines")
OUTPUT_DIR = Path("runs/analysis/mines")
OUTPUT_FILE = OUTPUT_DIR / "rare_classes.png"

RARE_CLASS_IDS = {0, 1, 3, 5, 7, 10, 11}
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(label_file: Path) -> Path | None:
    image_dir = label_file.parent.parent / "images"

    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{label_file.stem}{extension}"
        if candidate.exists():
            return candidate

    return None


def load_annotations(label_file: Path):
    annotations = []

    for line in label_file.read_text().splitlines():
        values = line.split()

        if not values:
            continue

        class_id = int(values[0])
        coords = list(map(float, values[1:]))

        if len(coords) == 4:
            # Bounding box YOLO:
            # x_center, y_center, width, height
            x, y, w, h = coords

        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # Segmentação YOLO:
            # x1, y1, x2, y2, ..., xn, yn
            xs = coords[0::2]
            ys = coords[1::2]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x = (x_min + x_max) / 2
            y = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

        else:
            raise ValueError(
                f"Anotação inválida em {label_file}: {line}"
            )

        annotations.append((class_id, x, y, w, h))

    return annotations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (DATASET_DIR / "data.yaml").open() as file:
        config = yaml.safe_load(file)

    names = config["names"]

    # Uma imagem pode conter várias anotações raras.
    images = defaultdict(list)
    total_rare_annotations = 0

    for split in SPLITS:
        labels_dir = DATASET_DIR / split / "labels"

        for label_file in sorted(labels_dir.glob("*.txt")):
            annotations = load_annotations(label_file)

            rare = [
                annotation
                for annotation in annotations
                if annotation[0] in RARE_CLASS_IDS
            ]

            if not rare:
                continue

            image_path = find_image(label_file)

            if image_path is None:
                print(f"Imagem não encontrada: {label_file}")
                continue

            images[(split, image_path)].extend(rare)
            total_rare_annotations += len(rare)

    if not images:
        print("Nenhuma anotação rara encontrada.")
        return

    print(f"Anotações raras encontradas: {total_rare_annotations}")
    print(f"Imagens correspondentes: {len(images)}")

    ncols = min(3, len(images))
    nrows = math.ceil(len(images) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, ((split, image_path), annotations) in zip(
        axes.flat, images.items()
    ):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            ax.imshow(image)

        for class_id, x, y, w, h in annotations:
            left = (x - w / 2) * width
            top = (y - h / 2) * height

            rect = patches.Rectangle(
                (left, top),
                w * width,
                h * height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )

            ax.add_patch(rect)

            ax.text(
                left,
                max(0, top - 5),
                f"{class_id}: {names[class_id]}",
                color="white",
                fontsize=9,
                bbox={
                    "facecolor": "red",
                    "alpha": 0.85,
                    "pad": 3,
                },
            )

        ax.set_title(
            f"{split}: {image_path.name}",
            fontsize=10,
        )

    fig.suptitle(
        "Mines Dataset — inspeção das classes raras",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Painel salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
