from pathlib import Path

from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "datasets" / "woodscape" / "woodscape.yaml"


def main():
    print("=" * 60)
    print("Treinamento RT-DETR - WoodScape")
    print("=" * 60)

    print(f"Dataset: {DATASET}")

    # Modelo pré-treinado no COCO
    model = RTDETR("rtdetr-l.pt")

    model.info()

    results = model.train(
        data=str(DATASET),

        # Smoke test inicial
        epochs=3,

        imgsz=640,

        batch=4,

        device=0,

        workers=8,

        seed=42,

        # Recomendado para RT-DETR em CUDA
        deterministic=False,

        project=str(ROOT / "runs" / "rtdetr"),

        name="woodscape_test",

        exist_ok=True,
    )

    print("\nTreinamento concluído.")

    return results


if __name__ == "__main__":
    main()