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

        # Warm-up / early stopping
        warmup_epochs=3,
        patience=20,

        # Transfer learning
        pretrained=True,

        # Reprodutibilidade
        seed=42,
        deterministic=False, # Recomendado para RT-DETR em CUDA

        # Mixed precision
        amp=True,

        # Checkpoints
        save=True,
        save_period=10,

         # Organização
        project=str(ROOT / "runs" / "rtdetr"),
        
        name="woodscape_test",
        
        exist_ok=False,
    )

    print("\nTreinamento concluído.")

    return results


if __name__ == "__main__":
    main()