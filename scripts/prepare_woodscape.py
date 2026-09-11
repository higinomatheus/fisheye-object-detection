import argparse
import csv
import random
import re
import shutil
from pathlib import Path

import yaml
from PIL import Image


CLASS_NAMES = {
    0: "vehicles",
    1: "person",
    2: "bicycle",
    3: "traffic_light",
    4: "traffic_sign",
}


CAMERA_PATTERN = re.compile(
    r"_(FV|RV|MVL|MVR)$",
    re.IGNORECASE
)


def get_capture_id(stem: str) -> str:
    """
    Exemplo:
        00001_FV  -> 00001
        00001_RV  -> 00001
        00002_MVL -> 00002

    Isso permite manter imagens da mesma captura
    dentro do mesmo split.
    """
    return CAMERA_PATTERN.sub("", stem)


def convert_annotation(
    annotation_path: Path,
    output_path: Path,
    image_width: int,
    image_height: int
):
    """
    Converte:

    WoodScape:
        class_name,class_id,xmin,ymin,xmax,ymax

    para YOLO:
        class_id x_center y_center width height
    """

    output_lines = []

    with annotation_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.reader(file)

        for row in reader:

            if not row or len(row) < 6:
                continue

            try:
                class_id = int(row[1].strip())

                xmin = float(row[2].strip())
                ymin = float(row[3].strip())
                xmax = float(row[4].strip())
                ymax = float(row[5].strip())

            except (ValueError, IndexError):
                print(
                    f"Anotação inválida em "
                    f"{annotation_path}: {row}"
                )
                continue

            if class_id not in CLASS_NAMES:
                print(
                    f"Classe desconhecida {class_id} "
                    f"em {annotation_path}"
                )
                continue

            # Garante que a bounding box permaneça
            # dentro da imagem
            xmin = max(0.0, min(xmin, image_width))
            xmax = max(0.0, min(xmax, image_width))

            ymin = max(0.0, min(ymin, image_height))
            ymax = max(0.0, min(ymax, image_height))

            if xmax <= xmin or ymax <= ymin:
                continue

            # Converte xmin/ymin/xmax/ymax para
            # center_x/center_y/width/height
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            center_x = xmin + bbox_width / 2.0
            center_y = ymin + bbox_height / 2.0

            # Normalização
            center_x /= image_width
            center_y /= image_height

            bbox_width /= image_width
            bbox_height /= image_height

            output_lines.append(
                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{bbox_width:.6f} "
                f"{bbox_height:.6f}"
            )

    output_path.write_text(
        "\n".join(output_lines)
        + ("\n" if output_lines else ""),
        encoding="utf-8"
    )

    return len(output_lines)


def create_image(
    source: Path,
    destination: Path,
    mode: str
):
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def create_directories(output_root: Path):

    for split in ["train", "val", "test"]:

        (output_root / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (output_root / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


def create_yaml(output_root: Path):

    config = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }

    yaml_path = output_root / "woodscape.yaml"

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"\nArquivo YAML criado em:")
    print(yaml_path)


def prepare_dataset(
    raw_root: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    mode: str
):

    images_dir = raw_root / "rgb_images"
    annotations_dir = raw_root / "box_2d_annotations"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Diretório não encontrado: {images_dir}"
        )

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"Diretório não encontrado: {annotations_dir}"
        )

    create_directories(output_root)

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg"
    }

    images = [
        file
        for file in images_dir.iterdir()
        if file.suffix.lower() in image_extensions
    ]

    annotations = {
        file.stem: file
        for file in annotations_dir.glob("*.txt")
    }

    # Mantém somente imagens com arquivo
    # de anotação correspondente
    pairs = []

    for image in images:

        annotation = annotations.get(image.stem)

        if annotation:
            pairs.append(
                (image, annotation)
            )

    print(f"Imagens encontradas: {len(images)}")
    print(f"Anotações encontradas: {len(annotations)}")
    print(f"Pares válidos: {len(pairs)}")

    if not pairs:
        raise RuntimeError(
            "Nenhum par imagem/anotação encontrado."
        )

    # --------------------------------------------------
    # Agrupa as quatro câmeras da mesma captura
    # --------------------------------------------------

    groups = {}

    for image, annotation in pairs:

        capture_id = get_capture_id(image.stem)

        groups.setdefault(
            capture_id,
            []
        ).append(
            (image, annotation)
        )

    capture_ids = list(groups.keys())

    random_generator = random.Random(seed)
    random_generator.shuffle(capture_ids)

    number_groups = len(capture_ids)

    train_end = int(
        number_groups * train_ratio
    )

    val_end = train_end + int(
        number_groups * val_ratio
    )

    train_groups = set(
        capture_ids[:train_end]
    )

    val_groups = set(
        capture_ids[train_end:val_end]
    )

    test_groups = set(
        capture_ids[val_end:]
    )

    split_groups = {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    object_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    # --------------------------------------------------
    # Processamento
    # --------------------------------------------------

    for split, selected_groups in split_groups.items():

        for capture_id in selected_groups:

            for image_path, annotation_path in groups[capture_id]:

                image_destination = (
                    output_root
                    / "images"
                    / split
                    / image_path.name
                )

                label_destination = (
                    output_root
                    / "labels"
                    / split
                    / f"{image_path.stem}.txt"
                )

                create_image(
                    image_path,
                    image_destination,
                    mode
                )

                with Image.open(image_path) as image:
                    width, height = image.size

                number_objects = convert_annotation(
                    annotation_path,
                    label_destination,
                    width,
                    height
                )

                split_counts[split] += 1
                object_counts[split] += number_objects

    create_yaml(output_root)

    print("\n======================================")
    print("WOODSCAPE PREPARADO")
    print("======================================")

    for split in ["train", "val", "test"]:
        print(
            f"{split:5s}: "
            f"{split_counts[split]:5d} imagens | "
            f"{object_counts[split]:6d} objetos"
        )

    print("======================================")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Diretório do WoodScape original"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Diretório de saída"
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--val",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="symlink",
        help=(
            "copy = copia imagens; "
            "symlink = cria links simbólicos"
        )
    )

    args = parser.parse_args()

    if args.train + args.val >= 1:
        raise ValueError(
            "train + val deve ser menor que 1."
        )

    prepare_dataset(
        raw_root=args.raw,
        output_root=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
        mode=args.mode
    )


if __name__ == "__main__":
    main()