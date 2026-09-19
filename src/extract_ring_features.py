"""Extract distance-aware ring features from omnidirectional-sonar frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_SIZE = (800, 600)
N_RINGS = 59
RING_WIDTH = 5


def read_calibration(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    required = {"ring_id", "pixels_per_fish"}
    if not required.issubset(table.columns):
        raise ValueError(f"Calibration file requires columns: {sorted(required)}")
    table = table.sort_values("ring_id")
    if table["ring_id"].tolist() != list(range(1, N_RINGS + 1)):
        raise ValueError(f"Calibration must contain ring_id 1 through {N_RINGS}")
    values = table["pixels_per_fish"].to_numpy(dtype=float)
    if np.any(values <= 0):
        raise ValueError("pixels_per_fish values must be positive")
    return values


def read_mask(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST) > 0


def ring_features(image_path: Path, calibration: np.ndarray,
                  cage_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    height, width = image.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((xx - width // 2) ** 2 + (yy - height // 2) ** 2)

    pixel_counts = np.zeros(N_RINGS, dtype=float)
    fish_equivalents = np.zeros(N_RINGS, dtype=float)
    for index in range(N_RINGS):
        inner = index * RING_WIDTH
        outer = (index + 1) * RING_WIDTH
        selected = (radius >= inner) & (radius < outer)
        if cage_mask is not None:
            selected &= cage_mask
        samples = image[selected]
        if samples.size == 0:
            continue
        threshold, _ = cv2.threshold(
            samples.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        pixel_counts[index] = np.count_nonzero(samples > threshold)
        fish_equivalents[index] = pixel_counts[index] / calibration[index]
    return pixel_counts, fish_equivalents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("calibration_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--aggregation", choices=("mean", "median", "sum"), default="mean")
    args = parser.parse_args()

    calibration = read_calibration(args.calibration_csv)
    cage_mask = read_mask(args.mask)
    images = sorted(
        p for p in args.image_dir.iterdir()
        if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if not images:
        raise ValueError(f"No sonar images found in {args.image_dir}")

    pixel_rows, fish_rows = [], []
    for image in images:
        pixels, fish = ring_features(image, calibration, cage_mask)
        pixel_rows.append([image.name, *pixels])
        fish_rows.append([image.name, *fish])

    columns = [f"ring_{index:02d}" for index in range(1, N_RINGS + 1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pixel_table = pd.DataFrame(pixel_rows, columns=["frame_id", *columns])
    fish_table = pd.DataFrame(fish_rows, columns=["frame_id", *columns])
    pixel_table.to_csv(args.output_dir / "ring_pixel_counts.csv", index=False)
    fish_table.to_csv(args.output_dir / "ring_fish_equivalents.csv", index=False)

    values = fish_table[columns]
    aggregate = getattr(values, args.aggregation)(axis=0)
    aggregate.to_frame().T.to_csv(args.output_dir / "dataset_ring_features.csv", index=False)


if __name__ == "__main__":
    main()

