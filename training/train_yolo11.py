from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "datasets" / "woodscape" / "woodscape.yaml"


def main():
    print("=" * 60)
    print("Treinamento YOLO11m - WoodScape")
    print("=" * 60)
    print(f"Dataset: {DATASET}")

    # Modelo YOLO11m pré-treinado no COCO
    model = YOLO("yolo11m.pt")

    model.info()

    results = model.train(
        # Dataset
        data=str(DATASET),

        # Treinamento
        epochs=100,
        imgsz=640,
        batch=8,

        # Hardware
        device=0,
        workers=8,

        # Otimização
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # Warm-up / Early Stopping
        warmup_epochs=3,
        patience=20,

        # Transfer Learning
        pretrained=True,

        # Reprodutibilidade
        seed=42,
        deterministic=False,

        # Mixed Precision
        amp=True,

        # Dataset
        cache=False,

        # Checkpoints
        save=True,
        save_period=10,

        # Organização dos resultados
        project=str(ROOT / "runs" / "yolo11"),
        name="woodscape_yolo11m_e100_b8",
        exist_ok=False,
    )

    print("\nTreinamento concluído.")

    return results


if __name__ == "__main__":
    main()