"""Exploratory HOG + PCA on WoodScape crops (not a final evaluation).

Run from repository root:
 python classical_ml/hog_pca.py --crops datasets/woodscape_crops_test --class-name person
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--crops', type=Path, required=True)
    ap.add_argument('--class-name', default='person')
    ap.add_argument('--components', type=int, default=10)
    ap.add_argument('--output', type=Path, default=None,
                    help='Optional output directory (default: results/e5_hog_pca_<components>)')
    args = ap.parse_args()
    manifest = args.crops / f'{args.class_name}_manifest.csv'
    if not manifest.exists():
        raise FileNotFoundError(f'Manifest not found: {manifest}')
    with manifest.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        raise ValueError('At least 3 crops are required')
    vectors, names = [], []
    for row in rows:
        p = args.crops / row['crop']
        with Image.open(p) as im:
            gray = im.convert('L').resize((64, 128))
            arr = np.asarray(gray, dtype=np.float32) / 255.0
        descriptor = hog(arr, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        vectors.append(descriptor)
        names.append(row['crop'])
    X = np.asarray(vectors, dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = min(args.components, X.shape[0] - 1, X.shape[1])
    if n_components < 1:
        raise ValueError('No valid PCA dimensions')
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    Z = pca.fit_transform(X_scaled)
    output = args.output or Path('results') / f'e5_hog_pca_{n_components}'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'exploratory_features.npz',
                        hog=X, pca=Z, names=np.asarray(names))
    explained = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, n_components + 1), explained, marker='o')
    ax.set(xlabel='Number of PCA components', ylabel='Cumulative explained variance',
           title='Exploratory HOG + PCA (same sample, no holdout)')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'explained_variance.png', dpi=160)
    plt.close(fig)
    print(f'Crops processed: {len(names)}')
    print(f'HOG dimensions: {X.shape[1]}')
    print(f'PCA dimensions: {n_components}')
    print(f'Cumulative explained variance: {explained[-1]:.4f}')
    print(f'Files saved to: {output}')
    print('EXPLORATORY ONLY: no train/val/test split, do not report as model performance.')


if __name__ == '__main__':
    main()
