"""
crop_utils.py
-------------
Shared adaptive mouth-cropping utilities used by every loader in the project.

This module is the single implementation of the "variable resolution" feature.
All loaders (GRID, MIRACL, custom) import from here, so the cropping logic
is identical across training and inference.

Design principles:
  - Crop is always derived from the ACTUAL dlib lip landmarks, scaled to the
    face size in the source video. A face that is 400px tall produces a larger
    crop than a face that is 100px tall — preserving the native detail.
  - CROP_PADDING_RATIO controls how much context beyond the tight lip bounding
    box is included. Default 0.5 means the crop is 2× the lip box on each axis.
  - All dimensions are rounded up to CROP_ALIGN_MULTIPLE (8) so MaxPool3D
    halving never produces fractional sizes inside the model.
  - GlobalAveragePooling2D in the model makes the CNN fully agnostic to the
    resulting crop dimensions.

Exports:
    crop_mouth(frame_rgb, landmarks) -> np.ndarray or None
        Returns the adaptive RGB crop for one frame.

    align_dim(value) -> int
        Round a pixel dimension up to the nearest CROP_ALIGN_MULTIPLE.

    canonical_size(frames) -> (H, W)
        Given a list of crops from one video, return the (H, W) that every
        crop in this video should be padded/resized to for batch compatibility.
        Uses the MEDIAN crop size across frames so one outlier frame can't
        blow up the batch memory.

    normalize_rgb(array) -> np.ndarray
        Per-channel normalization with fixed GRID statistics.
"""

import math
import numpy as np
import cv2

from config import (
    CHANNEL_MEANS, CHANNEL_STDS,
    CROP_ALIGN_MULTIPLE, CROP_PADDING_RATIO,
    MIN_CROP_H, MIN_CROP_W,
)

# Dlib mouth landmark indices (outer + inner lip contour)
MOUTH_POINTS = list(range(48, 68))


# ─────────────────────────────────────────────────────────────────────────────
# Dimension alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_dim(value: int) -> int:
    """
    Round value UP to the nearest CROP_ALIGN_MULTIPLE (default 8).

    Example: align_dim(53) -> 56   align_dim(64) -> 64
    """
    m = CROP_ALIGN_MULTIPLE
    return int(math.ceil(value / m) * m)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_rgb(frames: np.ndarray) -> np.ndarray:
    """
    Apply per-channel normalization with fixed GRID training-set statistics.

    Paper Appendix A.2:
        muR=0.7136 sigR=0.1138  muG=0.4906 sigG=0.1078  muB=0.3283 sigB=0.0917

    Accepts any shape (..., H, W, 3). Values may be uint8 [0,255] or
    float [0,1] — both are handled.

    Returns float32 array of same shape.
    """
    arr = np.array(frames, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    means = np.array(CHANNEL_MEANS, dtype=np.float32)
    stds  = np.array(CHANNEL_STDS,  dtype=np.float32)
    return (arr - means) / stds


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame adaptive crop
# ─────────────────────────────────────────────────────────────────────────────

def crop_mouth(frame_rgb: np.ndarray, landmarks) -> np.ndarray:
    """
    Crop the mouth region from a single RGB frame using dlib landmarks.

    The crop adapts to the face size in the video:
      - A face 400px wide → large crop in pixels (preserves detail)
      - A face 100px wide → smaller crop (no upscaling artefacts)

    Steps:
      1. Collect the 20 mouth landmark coordinates (indices 48–67).
      2. Compute the tight bounding box around them.
      3. Expand by CROP_PADDING_RATIO on every side.
      4. Clamp to frame boundaries.
      5. Align height and width to CROP_ALIGN_MULTIPLE.
      6. Return the RGB crop (no resize — native resolution).

    Args:
        frame_rgb:  H×W×3 uint8 RGB array.
        landmarks:  dlib full_object_detection (68 landmarks).

    Returns:
        np.ndarray crop of shape (crop_H, crop_W, 3) uint8.
        Returns a MIN_CROP_H × MIN_CROP_W black frame if the crop is empty.
    """
    fh, fw = frame_rgb.shape[:2]

    # Mouth landmark coordinates
    coords = np.array([
        [landmarks.part(n).x, landmarks.part(n).y]
        for n in MOUTH_POINTS
    ], dtype=np.float32)

    # Tight bounding box
    x_min_tight = coords[:, 0].min()
    x_max_tight = coords[:, 0].max()
    y_min_tight = coords[:, 1].min()
    y_max_tight = coords[:, 1].max()

    lip_w = x_max_tight - x_min_tight
    lip_h = y_max_tight - y_min_tight

    # Add proportional padding
    pad_x = lip_w * CROP_PADDING_RATIO
    pad_y = lip_h * CROP_PADDING_RATIO

    x_min = int(max(x_min_tight - pad_x, 0))
    x_max = int(min(x_max_tight + pad_x, fw))
    y_min = int(max(y_min_tight - pad_y, 0))
    y_max = int(min(y_max_tight + pad_y, fh))

    crop_h = y_max - y_min
    crop_w = x_max - x_min

    if crop_h < 1 or crop_w < 1:
        return np.zeros((MIN_CROP_H, MIN_CROP_W, 3), dtype=np.uint8)

    crop = frame_rgb[y_min:y_max, x_min:x_max]

    # Enforce minimum size
    if crop_h < MIN_CROP_H or crop_w < MIN_CROP_W:
        new_h = max(crop_h, MIN_CROP_H)
        new_w = max(crop_w, MIN_CROP_W)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        crop_h, crop_w = new_h, new_w

    # Align dimensions to CROP_ALIGN_MULTIPLE
    aligned_h = align_dim(crop_h)
    aligned_w = align_dim(crop_w)

    if aligned_h != crop_h or aligned_w != crop_w:
        crop = cv2.resize(crop, (aligned_w, aligned_h), interpolation=cv2.INTER_LANCZOS4)

    return crop  # dtype uint8, shape (aligned_H, aligned_W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical batch size for one video
# ─────────────────────────────────────────────────────────────────────────────

def canonical_size(crops: list) -> tuple:
    """
    Compute a single (H, W) that all crops from one video will be resized to
    before stacking into a tensor.

    Uses the MEDIAN height and width across all detected frames, then aligns
    to CROP_ALIGN_MULTIPLE. This is robust to:
      - Occasional missed detections returning the MIN_CROP blank frame.
      - One frame where the person leans close to the camera.
      - Variable face size across a long clip.

    Args:
        crops: list of np.ndarray crops, each shape (h_i, w_i, 3).

    Returns:
        (H, W) tuple of ints — aligned to CROP_ALIGN_MULTIPLE.
    """
    heights = [c.shape[0] for c in crops if c is not None and c.size > 0]
    widths  = [c.shape[1] for c in crops if c is not None and c.size > 0]

    if not heights:
        return (MIN_CROP_H, MIN_CROP_W)

    med_h = int(np.median(heights))
    med_w = int(np.median(widths))

    return (align_dim(max(med_h, MIN_CROP_H)),
            align_dim(max(med_w, MIN_CROP_W)))


# ─────────────────────────────────────────────────────────────────────────────
# Resize a list of variable-size crops to one canonical size
# ─────────────────────────────────────────────────────────────────────────────

def resize_to_canonical(crops: list, size: tuple) -> np.ndarray:
    """
    Resize all crops in a list to (size[0], size[1]) and stack into an array.

    Uses LANCZOS4 for downscaling (better quality than default INTER_LINEAR)
    and LANCZOS4 for upscaling too (smoother than INTER_CUBIC for faces).

    Args:
        crops: list of np.ndarray, each (h_i, w_i, 3) uint8.
        size:  (H, W) target size.

    Returns:
        np.ndarray of shape (N, H, W, 3) uint8.
    """
    H, W = size
    result = []
    for crop in crops:
        if crop.shape[:2] == (H, W):
            result.append(crop)
        else:
            resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4)
            result.append(resized)
    return np.stack(result, axis=0)
