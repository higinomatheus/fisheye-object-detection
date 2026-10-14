#!/usr/bin/env python3
"""Cria comparações lado a lado dos vídeos anotados por três detectores.

Dependência: opencv-python (cv2).
Executar a partir de qualquer diretório, após colocar este script em evaluation/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolo11s", "yolo11m", "rtdetr_l")
TITLES = ("YOLO11s", "YOLO11m", "RT-DETR-L")
DEFAULT_SCENES = ("mineracao", "fisheye_city")


def find_video(directory: Path, scene: str) -> Path:
    """Encontra um vídeo anotado em MP4 ou AVI, sem escolher ambiguamente."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Diretório ausente: {directory}")

    for extension in (".mp4", ".avi"):
        preferred = directory / f"{scene}{extension}"
        if preferred.is_file():
            return preferred

    videos = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi"}
    )
    if len(videos) == 1:
        return videos[0]
    if not videos:
        raise FileNotFoundError(f"Nenhum arquivo MP4 ou AVI encontrado em: {directory}")
    raise RuntimeError(f"Vários vídeos em {directory}: {videos}. Mantenha somente um ou renomeie para {scene}.mp4/.avi")


def prepare_tile(frame, tile_width: int, tile_height: int):
    h, w = frame.shape[:2]
    scale = min(tile_width / w, tile_height / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    import numpy as np
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    x = (tile_width - new_w) // 2
    y = (tile_height - new_h) // 2
    tile[y:y+new_h, x:x+new_w] = resized
    return tile


def compare(scene: str, root: Path, output_dir: Path, tile_width: int, fps_tolerance: float):
    paths = [find_video(root / "runs" / "detect" / "video" / f"{model}_{scene}", scene) for model in MODELS]
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    writer = None
    try:
        if not all(cap.isOpened() for cap in captures):
            raise RuntimeError("Não foi possível abrir um ou mais vídeos: " + ", ".join(map(str, paths)))
        fps = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
        counts = [round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
        dims = [(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for cap in captures]
        print(f"\nCena: {scene}")
        for model, path, rate, count, dim in zip(MODELS, paths, fps, counts, dims):
            print(f"  {model}: {path} | {dim[0]}x{dim[1]} | {rate:.3f} FPS | {count} frames")
        if any(rate <= 0 for rate in fps):
            raise ValueError("FPS inválido: verifique os arquivos de vídeo")
        if max(fps) - min(fps) > fps_tolerance:
            raise ValueError("Os vídeos possuem FPS diferentes; não é seguro sincronizar somente por frame")
        if len(set(counts)) != 1:
            raise ValueError("Os vídeos possuem quantidades diferentes de frames; revise os arquivos antes de combinar")
        if counts[0] <= 0:
            raise ValueError("Quantidade de frames inválida")

        tile_height = round(tile_width * dims[0][1] / dims[0][0])
        header_height = 54
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"comparison_{scene}.mp4"
        if target.exists():
            raise FileExistsError(f"Saída já existe: {target} (mova ou remova antes de executar novamente)")
        width, height = 3 * tile_width, tile_height + header_height
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps[0], (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Não foi possível criar o MP4: {target}")

        import numpy as np
        for frame_id in range(counts[0]):
            frames = [cap.read() for cap in captures]
            if not all(ok for ok, _ in frames):
                raise RuntimeError(f"Falha de leitura no frame {frame_id + 1}; vídeo parcial em {target}")
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for index, ((_, frame), title) in enumerate(zip(frames, TITLES)):
                x = index * tile_width
                canvas[header_height:, x:x + tile_width] = prepare_tile(frame, tile_width, tile_height)
                cv2.putText(canvas, title, (x + 18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                if index:
                    cv2.line(canvas, (x, 0), (x, height), (110, 110, 110), 2)
            writer.write(canvas)
            if (frame_id + 1) % 200 == 0:
                print(f"  {frame_id + 1}/{counts[0]} frames")
        print(f"  Gerado: {target}")
    finally:
        for cap in captures:
            cap.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description="Compara YOLO11s, YOLO11m e RT-DETR-L em vídeos sincronizados")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES), help="Cenas (padrão: mineracao fisheye_city)")
    parser.add_argument("--root", type=Path, default=ROOT, help="Raiz do projeto")
    parser.add_argument("--output-dir", type=Path, default=None, help="Pasta dos vídeos comparativos")
    parser.add_argument("--tile-width", type=int, default=640, help="Largura de cada painel em pixels")
    parser.add_argument("--fps-tolerance", type=float, default=0.05, help="Diferença máxima de FPS tolerada")
    args = parser.parse_args()
    if args.tile_width < 160:
        parser.error("--tile-width deve ser >= 160")
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or root / "runs" / "comparison" / "video").expanduser().resolve()
    for scene in args.scenes:
        compare(scene, root, output_dir, args.tile_width, args.fps_tolerance)


if __name__ == "__main__":
    main()