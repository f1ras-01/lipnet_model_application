"""
data_loader.py
--------------
Video and alignment loading — fully adaptive spatial resolution.

Every loader now produces crops at native resolution (no fixed resize).
The spatial dimensions vary per video and are handled by:
  - crop_utils.canonical_size()  to pick one H×W per video
  - crop_utils.resize_to_canonical()  to standardize within a video
  - dataset.py padded_batch  to standardize within a training batch
  - model GlobalAveragePooling2D  to collapse spatial dims in the model

Three loaders:
  load_video(path, augment=False)
      GRID corpus .mpg -> (75, H, W, 3) float32   H,W vary by video

  load_alignments(path)
      GRID .align -> int64 character indices

  load_data(path, augment=False)
      Combines both — used via mappable_function in the tf.data pipeline

  load_custom_video(video_path)
      Any webcam/recorded video -> (75, H, W, 3) float32 via dlib
"""

import os
import random
import threading

import cv2
import dlib
import numpy as np
import tensorflow as tf

from config import (
    ALIGN_DIR, DATA_DIR, DLIB_MODEL_PATH,
    GRID_CROP_X, GRID_CROP_Y,
    MIN_CROP_H, MIN_CROP_W,
    TARGET_C, TARGET_FRAMES,
)
from crop_utils import (
    canonical_size, crop_mouth, normalize_rgb, resize_to_canonical,
)
from utils import char_to_num


# ─────────────────────────────────────────────────────────────────────────────
# Frame jitter augmentation (paper Section 4.1, p=0.05 per frame)
# ─────────────────────────────────────────────────────────────────────────────

def _frame_jitter(frames: list, p: float = 0.05) -> list:
    result = []
    for f in frames:
        r = random.random()
        if r < p / 2:
            continue
        elif r < p:
            result.extend([f, f])
        else:
            result.append(f)
    return result if result else frames


# ─────────────────────────────────────────────────────────────────────────────
# Dlib tools — thread-safe lazy initialization
#
# tf.data AUTOTUNE runs map() in parallel threads. Without a lock:
#   Thread A sets _detector but hasn't set _predictor yet.
#   Thread B sees _detector is not None, skips init, returns (detector, None).
#   predictor(gray, face) → TypeError: NoneType is not callable.
#
# Double-checked locking: fast check outside lock, guaranteed re-check inside.
# ─────────────────────────────────────────────────────────────────────────────

_dlib_lock = threading.Lock()
_detector  = None
_predictor = None


def _get_dlib():
    """Return (detector, predictor), initializing both atomically."""
    global _detector, _predictor
    if _detector is None or _predictor is None:
        with _dlib_lock:
            if _detector is None or _predictor is None:   # re-check under lock
                if not os.path.exists(DLIB_MODEL_PATH):
                    raise FileNotFoundError(
                        f"dlib model not found: {DLIB_MODEL_PATH}\n"
                        "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                    )
                _detector  = dlib.get_frontal_face_detector()
                _predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
    return _detector, _predictor


# ─────────────────────────────────────────────────────────────────────────────
# GRID corpus loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_video(path: str, augment: bool = False) -> np.ndarray:
    """
    Load a GRID .mpg file and return adaptively-cropped, normalized RGB frames.

    GRID has a fixed camera so a coarse pixel crop is applied first to isolate
    the lower-face area, then dlib refines the exact lip region within that area.
    This two-stage approach is faster than running dlib on the full frame and
    produces cleaner crops on the fixed-camera GRID setup.

    The output H×W varies per video (face distance, expression, recording session).

    Args:
        path:    Path to a GRID .mpg file.
        augment: If True, apply H-flip and frame jitter (training only).

    Returns:
        np.ndarray (75, H, W, 3) float32.  H and W are multiples of 8.
    """
    detector, predictor = _get_dlib()

    cap  = cv2.VideoCapture(path)
    raw_crops = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Stage 1: coarse pixel crop (GRID-specific fixed camera) ──────────
        # Extract the lower-face region to speed up dlib and reduce FP detections
        roi = frame_rgb[GRID_CROP_Y.start - 60 : GRID_CROP_Y.stop + 60,
                        GRID_CROP_X.start - 40 : GRID_CROP_X.stop + 40]

        # ── Stage 2: dlib adaptive lip crop ──────────────────────────────────
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        faces    = detector(gray_roi)

        if len(faces) == 0:
            # Fallback: use the fixed pixel crop directly at minimum size
            fallback = frame_rgb[GRID_CROP_Y, GRID_CROP_X]
            crop = cv2.resize(fallback, (MIN_CROP_W, MIN_CROP_H),
                              interpolation=cv2.INTER_LANCZOS4)
        else:
            face      = faces[0]
            landmarks = predictor(gray_roi, face)
            crop      = crop_mouth(roi, landmarks)

        raw_crops.append(crop)

    cap.release()

    # ── Augmentation ─────────────────────────────────────────────────────────
    if augment:
        raw_crops = _frame_jitter(raw_crops, p=0.05)
        if random.random() < 0.5:
            raw_crops = [np.fliplr(c) for c in raw_crops]

    # ── Fix frame count ───────────────────────────────────────────────────────
    total = len(raw_crops)
    if total == 0:
        dummy = np.zeros((MIN_CROP_H, MIN_CROP_W, TARGET_C), dtype=np.uint8)
        raw_crops = [dummy] * TARGET_FRAMES
    elif total < TARGET_FRAMES:
        raw_crops = raw_crops + [raw_crops[-1]] * (TARGET_FRAMES - total)
    else:
        idx       = np.linspace(0, total - 1, TARGET_FRAMES, dtype=int)
        raw_crops = [raw_crops[i] for i in idx]

    # ── Standardize spatial dims within this video ────────────────────────────
    # All 75 crops from one video are resized to the same (H, W) so they can
    # be stacked. padded_batch will then handle variation BETWEEN videos.
    size   = canonical_size(raw_crops)
    arr    = resize_to_canonical(raw_crops, size)    # (75, H, W, 3) uint8
    return normalize_rgb(arr)                        # (75, H, W, 3) float32


def load_alignments(path: str) -> tf.Tensor:
    """Parse GRID .align file -> integer-encoded character tensor."""
    with open(path, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if parts[2] != "sil":
            tokens = [*tokens, " ", parts[2]]
    return char_to_num(
        tf.reshape(tf.strings.unicode_split(tokens, input_encoding="UTF-8"), (-1,))
    )[1:]


def load_data(path: tf.Tensor, augment: bool = False):
    path_str   = bytes.decode(path.numpy())
    file_name  = path_str.replace("\\", "/").split("/")[-1].split(".")[0]
    video_path = os.path.join(DATA_DIR,  f"{file_name}.mpg")
    align_path = os.path.join(ALIGN_DIR, f"{file_name}.align")
    frames     = load_video(video_path, augment=augment)
    alignments = load_alignments(align_path)
    return frames, alignments


def mappable_function(path: str):
    return tf.py_function(
        lambda p: load_data(p, augment=False), [path], (tf.float32, tf.int64)
    )


def mappable_function_augment(path: str):
    return tf.py_function(
        lambda p: load_data(p, augment=True), [path], (tf.float32, tf.int64)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Custom video loader (your own recordings)
# ─────────────────────────────────────────────────────────────────────────────

def load_custom_video(video_path: str) -> np.ndarray:
    """
    Load any recorded video, detect the mouth adaptively, and return
    normalized frames at the video's native resolution.

    No fixed target H×W. The crop size is determined entirely by dlib —
    if the face is close to the camera you get a large, detailed crop;
    if the face is far you get a smaller crop. Both are handled correctly
    by the model's GlobalAveragePooling2D layer.

    Args:
        video_path: Path to any .mp4 / .mpg / .avi / .mov file.

    Returns:
        np.ndarray (75, H, W, 3) float32.  H, W are multiples of 8.

    Raises:
        FileNotFoundError: if dlib model is missing.
        ValueError:        if the video cannot be opened or has no frames.
    """
    detector, predictor = _get_dlib()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Read video metadata for progress reporting
    total_frames_approx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 25
    h_native   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_native   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"  Video info: {w_native}×{h_native} px  |  {fps_native:.1f} fps  |  ~{total_frames_approx} frames")

    raw_crops      = []
    detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces     = detector(gray)

        if len(faces) == 0:
            crop = None   # placeholder — will be filled after loop
        else:
            face      = faces[0]
            landmarks = predictor(gray, face)
            crop      = crop_mouth(frame_rgb, landmarks)
            detected_count += 1

        raw_crops.append(crop)

    cap.release()

    total = len(raw_crops)
    if total == 0:
        raise ValueError(f"No frames read from: {video_path}")

    detection_rate = detected_count / total * 100
    print(f"  Detected face in {detected_count}/{total} frames ({detection_rate:.1f}%)")
    if detection_rate < 50:
        print(f"  ⚠ Low detection rate. Try recording in better lighting, closer to camera.")

    # Fill None placeholders with the nearest detected crop
    # (forward-fill then backward-fill)
    last_good = None
    for i in range(len(raw_crops)):
        if raw_crops[i] is not None:
            last_good = raw_crops[i]
        elif last_good is not None:
            raw_crops[i] = last_good.copy()

    last_good = None
    for i in range(len(raw_crops) - 1, -1, -1):
        if raw_crops[i] is not None:
            last_good = raw_crops[i]
        elif last_good is not None:
            raw_crops[i] = last_good.copy()

    # Final fallback: still None means no face found anywhere
    if any(c is None for c in raw_crops):
        dummy = np.zeros((MIN_CROP_H, MIN_CROP_W, TARGET_C), dtype=np.uint8)
        raw_crops = [c if c is not None else dummy for c in raw_crops]

    # Fix frame count
    if total < TARGET_FRAMES:
        raw_crops = raw_crops + [raw_crops[-1]] * (TARGET_FRAMES - total)
    else:
        idx       = np.linspace(0, total - 1, TARGET_FRAMES, dtype=int)
        raw_crops = [raw_crops[i] for i in idx]

    # Standardize spatial dims within this video
    size = canonical_size(raw_crops)
    arr  = resize_to_canonical(raw_crops, size)   # (75, H, W, 3) uint8

    print(f"  Crop size  : {arr.shape[2]}×{arr.shape[1]} px  (adaptive, native resolution)")

    return normalize_rgb(arr)   # (75, H, W, 3) float32


if __name__ == "__main__":
    import glob
    mpg_files = glob.glob(os.path.join(DATA_DIR, "*.mpg"))
    if mpg_files:
        f, a = load_data(tf.constant(mpg_files[0]))
        print(f"frames: {f.shape}  |  min={f.min():.3f}  max={f.max():.3f}")
        print(f"labels: {a.shape}")
    else:
        print(f"No .mpg files in {DATA_DIR}")
