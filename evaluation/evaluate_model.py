from argparse import ArgumentParser
from pathlib import Path

from ultralytics import RTDETR, YOLO


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "datasets" / "woodscape" / "woodscape.yaml"


def load_model(model_type: str, weights: Path):
    if model_type == "rtdetr":
        return RTDETR(str(weights))

    if model_type == "yolo":
        return YOLO(str(weights))

    raise ValueError(f"Tipo de modelo não suportado: {model_type}")


def evaluate(
    model_type: str,
    weights: Path,
    name: str,
):
    print("=" * 70)
    print("Avaliação no conjunto de TESTE")
    print("=" * 70)

    print(f"Modelo:      {model_type}")
    print(f"Pesos:       {weights}")
    print(f"Dataset:     {DATASET}")
    print(f"Experimento: {name}")

    if not weights.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {weights}"
        )

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATASET}"
        )

    model = load_model(
        model_type=model_type,
        weights=weights,
    )

    metrics = model.val(
        data=str(DATASET),

        # IMPORTANTE:
        # utiliza somente o conjunto de teste
        split="test",

        imgsz=640,
        batch=8,

        device=0,
        workers=8,

        # Resultados
        plots=True,

        project=str(ROOT / "runs" / "evaluation"),
        name=name,

        exist_ok=False,
    )

    print("\n" + "=" * 70)
    print("RESULTADOS DO CONJUNTO DE TESTE")
    print("=" * 70)

    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50:95:  {metrics.box.map:.4f}")

    print("\nVelocidade:")

    for step, value in metrics.speed.items():
        print(f"{step}: {value:.4f} ms/imagem")

    print("\nResultados salvos em:")
    print(metrics.save_dir)

    return metrics


def main():
    parser = ArgumentParser(
        description=(
            "Avalia modelos YOLO ou RT-DETR "
            "no conjunto de teste do WoodScape."
        )
    )

    parser.add_argument(
        "--model-type",
        choices=["yolo", "rtdetr"],
        required=True,
        help="Tipo do modelo.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Caminho para o arquivo best.pt.",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Nome do experimento.",
    )

    args = parser.parse_args()

    evaluate(
        model_type=args.model_type,
        weights=args.weights,
        name=args.name,
    )


if __name__ == "__main__":
    main()