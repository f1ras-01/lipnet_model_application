"""
data_loader.py
--------------
All functions responsible for turning raw files on disk into tensors/arrays
that the model can consume.

Three loaders are provided:

  load_video(path)
      For GRID corpus .mpg files — uses a hardcoded pixel crop because
      the GRID corpus has a fixed camera with the speaker always in the
      same position in the frame.

  load_alignments(path)
      Reads a GRID .align file and returns integer-encoded character labels.

  load_data(path)
      Combines load_video + load_alignments into a single (frames, labels) tuple.
      Wrapped in a tf.py_function by mappable_function so it works inside a
      tf.data pipeline.

  load_custom_video(video_path)
      For your own recorded videos — uses dlib to detect and crop the mouth
      region dynamically, then pads/samples to exactly 75 frames.
"""

import os
from typing import List

import cv2
import dlib
import numpy as np
import tensorflow as tf

from config import (
    ALIGN_DIR, DATA_DIR, DLIB_MODEL_PATH,
    GRID_CROP_X, GRID_CROP_Y,
    TARGET_FRAMES, TARGET_H, TARGET_W,
)
from utils import char_to_num


# ─────────────────────────────────────────────────────────────────────────────
# GRID corpus loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_video(path: str) -> tf.Tensor:
    """
    Load a GRID corpus .mpg file and return normalised grayscale frames.

    Steps:
      1. Read every frame from the video with OpenCV.
      2. Convert to grayscale.
      3. Crop the fixed mouth region (190:236, 80:220) → shape (46, 140, 1).
      4. Z-normalise: subtract mean, divide by std.

    Returns:
        tf.Tensor of shape (N_frames, 46, 140, 1) dtype float32.
    """
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(frame)           # (H, W, 1)
        frames.append(frame[GRID_CROP_Y, GRID_CROP_X, :]) # (46, 140, 1)

    cap.release()

    mean = tf.math.reduce_mean(frames)
    std  = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> tf.Tensor:
    """
    Parse a GRID .align file and return integer-encoded character labels.

    .align format (one word per line):
        <start_frame> <end_frame> <word>
    Lines where the word is 'sil' (silence) are skipped.

    Returns:
        tf.Tensor of shape (N_chars,) dtype int64.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        parts = line.split()
        if parts[2] != "sil":
            tokens = [*tokens, " ", parts[2]]   # space-separated words

    # Split each word string into individual characters, flatten, encode
    return char_to_num(
        tf.reshape(
            tf.strings.unicode_split(tokens, input_encoding="UTF-8"),
            (-1,)
        )
    )[1:]   # drop the leading space


def load_data(path: tf.Tensor):
    """
    Combines load_video + load_alignments for one sample.

    Derives both the video path and the alignment path from the file stem
    (e.g. 'bbal6n' → data/s1/bbal6n.mpg + data/alignments/s1/bbal6n.align).

    Used via mappable_function inside the tf.data pipeline.

    Returns:
        (frames, alignments) — a (float32 tensor, int64 tensor) tuple.
    """
    path = bytes.decode(path.numpy())

    # Windows path splitting (change to '/' for Linux/Mac)
    file_name = path.split("\\")[-1].split(".")[0]
    # Fallback: also handle forward-slash paths
    if "/" in path and "\\" not in path:
        file_name = path.split("/")[-1].split(".")[0]

    video_path     = os.path.join(DATA_DIR,  f"{file_name}.mpg")
    alignment_path = os.path.join(ALIGN_DIR, f"{file_name}.align")

    frames     = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments


def mappable_function(path: str) -> List[str]:
    """
    Wraps load_data in tf.py_function so it can be used with dataset.map().

    tf.data requires functions that return tf.Tensors; tf.py_function is the
    bridge that lets plain Python functions run inside the pipeline.
    """
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Custom video loader (for your own recordings)
# ─────────────────────────────────────────────────────────────────────────────


def load_custom_video(
    video_path: str,
    target_frames: int = TARGET_FRAMES,
    target_h: int = TARGET_H,
    target_w: int = TARGET_W,
) -> np.ndarray:
    """
    Load a custom (non-GRID) video, detect the mouth region with dlib,
    crop and resize each frame, pad/sample to exactly 75 frames.

    Unlike load_video() which relies on a fixed pixel crop, this function
    dynamically detects facial landmarks in every frame so it works with
    any camera angle, resolution, or face position.

    Args:
        video_path:    Path to any .mp4 / .mpg / .avi / .mov file.
        target_frames: Number of output frames (default 75).
        target_h:      Output frame height in pixels (default 46).
        target_w:      Output frame width in pixels (default 140).

    Returns:
        np.ndarray of shape (75, 46, 140, 1) dtype float32, values in [0, 1].

    Raises:
        FileNotFoundError: if DLIB_MODEL_PATH does not exist.
        ValueError:        if the video cannot be opened or has no frames.
    """

    # ── 0. Guard: check dlib model exists ────────────────────────────────────
    if not os.path.exists(DLIB_MODEL_PATH):
        raise FileNotFoundError(
            f"dlib landmark model not found at: {DLIB_MODEL_PATH}\n"
            "Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )

    # ── 1. Load dlib detector and landmark predictor ──────────────────────────
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

    # Landmark indices 48–67 correspond to the outer and inner lip contours
    MOUTH_POINTS = list(range(48, 68))

    # ── 2. Open the video ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            # No face detected — use a black (zero) frame as a placeholder
            mouth_resized = np.zeros((target_h, target_w), dtype=np.uint8)
        else:
            face      = faces[0]   # use the first / largest detected face
            landmarks = predictor(gray, face)

            # Collect (x, y) of every mouth landmark
            mouth_coords = np.array([
                [landmarks.part(n).x, landmarks.part(n).y]
                for n in MOUTH_POINTS
            ])

            # Bounding box with 10-pixel padding on all sides
            x_min = max(int(mouth_coords[:, 0].min()) - 10, 0)
            x_max = min(int(mouth_coords[:, 0].max()) + 10, frame.shape[1])
            y_min = max(int(mouth_coords[:, 1].min()) - 10, 0)
            y_max = min(int(mouth_coords[:, 1].max()) + 10, frame.shape[0])

            mouth_crop    = gray[y_min:y_max, x_min:x_max]
            # cv2.resize takes (width, height) — note the order!
            mouth_resized = cv2.resize(mouth_crop, (target_w, target_h))

        frames.append(mouth_resized)

    cap.release()

    # ── 3. Normalise frame count to exactly target_frames ────────────────────
    total = len(frames)
    if total == 0:
        raise ValueError(f"No frames could be read from: {video_path}")

    if total < target_frames:
        # Pad by repeating the last frame
        pad    = [frames[-1]] * (target_frames - total)
        frames = frames + pad
    else:
        # Uniformly sub-sample so we keep target_frames evenly spread frames
        indices = np.linspace(0, total - 1, target_frames, dtype=int)
        frames  = [frames[i] for i in indices]

    # ── 4. Stack, add channel dim, normalise to [0, 1] ───────────────────────
    video_array = np.stack(frames, axis=0)               # (75, 46, 140)
    video_array = np.expand_dims(video_array, axis=-1)   # (75, 46, 140, 1)
    # ---Z-normalize: subtract mean, divide by std ---
    # This MUST match what load_video() does for GRID training data.
    # The model weights were tuned for inputs with mean≈0, std≈1.
    video_array = video_array.astype(np.float32)
    mean = video_array.mean()
    std  = video_array.std()
    if std > 0:
        video_array = (video_array - mean) / std
    else:
        video_array = video_array - mean   # fallback: avoid division by zero

    return video_array


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run this file directly to verify loaders work)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    # Test GRID loader on the first available .mpg file
    mpg_files = glob.glob(os.path.join(DATA_DIR, "*.mpg"))
    if mpg_files:
        test_file = mpg_files[0]
        print(f"Testing GRID loader on: {test_file}")
        frames, aligns = load_data(tf.constant(test_file))
        print(f"  frames shape : {frames.shape}")
        print(f"  aligns shape : {aligns.shape}")
    else:
        print(f"No .mpg files found in {DATA_DIR} — skipping GRID loader test.")
