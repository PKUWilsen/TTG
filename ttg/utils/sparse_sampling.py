from __future__ import annotations

from typing import Optional
import numpy as np
import torch


def random_sample_1channel(dense_image: np.ndarray, num_points: int, fill_value: float = 0.0) -> np.ndarray:
    """Randomly sample pixels from a single-channel dense radio map.

    This mirrors the original TTG implementation: unobserved pixels are filled
    with zeros before the common [-1, 1] transform, which turns them into -1.
    """
    if dense_image.ndim != 2:
        raise ValueError(f"Expected a 2D dense image, got shape {dense_image.shape}")
    h, w = dense_image.shape
    total = h * w
    flat_indices = np.random.choice(total, int(num_points), replace=False)
    rows = flat_indices // h
    cols = flat_indices % h
    sparse = np.full((h, w), fill_value)
    sparse[rows, cols] = dense_image[rows, cols]
    return sparse


def detect_sift_points(img_tensor_minus1_1: torch.Tensor, image_size: int) -> np.ndarray:
    """Detect SIFT keypoints on a normalized image tensor and return flattened indices."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for SIFT-based sparse enhancement. Install opencv-contrib-python.") from exc

    arr = img_tensor_minus1_1.detach().cpu().squeeze().numpy()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got shape {arr.shape}")
    height, width = arr.shape
    _ = height
    gray = (((arr + 1.0) / 2.0) * 255.0).astype(np.uint8)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    if not keypoints:
        return np.array([], dtype=np.int64)
    indices = [int(p.pt[1]) * width + int(p.pt[0]) for p in keypoints]
    return np.unique(np.asarray(indices, dtype=np.int64))


def create_enhanced_sparse_map_sift(
    truth_sparse_map_np: np.ndarray,
    coarse_map_tensor: torch.Tensor,
    num_true_points: int,
    num_total_points: int,
    image_size: int,
) -> np.ndarray:
    """Add coarse-map values at SIFT-selected unknown positions to form enhanced sparse conditioning."""
    enhanced = np.full_like(truth_sparse_map_np, -1.0, dtype=np.float32)
    true_mask = truth_sparse_map_np != -1.0
    enhanced[true_mask] = truth_sparse_map_np[true_mask]

    num_to_sample = int(num_total_points) - int(num_true_points)
    if num_to_sample <= 0:
        return enhanced

    unknown_indices = np.where(truth_sparse_map_np.reshape(-1) == -1.0)[0]
    if unknown_indices.size == 0:
        return enhanced

    sift_indices = detect_sift_points(coarse_map_tensor, image_size=image_size)
    smart_indices = np.intersect1d(sift_indices, unknown_indices, assume_unique=True)

    selected: list[int] = []
    if smart_indices.size >= num_to_sample:
        selected.extend(np.random.choice(smart_indices, num_to_sample, replace=False).tolist())
    else:
        selected.extend(smart_indices.tolist())
        remaining = num_to_sample - len(selected)
        other_indices = np.setdiff1d(unknown_indices, smart_indices, assume_unique=True)
        if other_indices.size >= remaining:
            selected.extend(np.random.choice(other_indices, remaining, replace=False).tolist())

    if selected:
        coarse_flat = coarse_map_tensor.detach().cpu().squeeze().numpy().reshape(-1)
        enhanced_flat = enhanced.reshape(-1)
        enhanced_flat[np.asarray(selected, dtype=np.int64)] = coarse_flat[np.asarray(selected, dtype=np.int64)]
        enhanced = enhanced_flat.reshape(enhanced.shape)
    return enhanced.astype(np.float32)
