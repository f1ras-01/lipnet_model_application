"""
data_loader_miracl.py
---------------------
Adapter that makes the MIRACL-VC1 dataset compatible with your LipNet model.

MIRACL-VC1 folder structure (what you download from Kaggle):
    miraclvc1/
    ├── F01/                    ← speaker ID (F = female, M = male)
    │   ├── words/
    │   │   ├── 01/             ← word ID (01–10)
    │   │   │   ├── 01/         ← instance number (01–10)
    │   │   │   │   ├── color/
    │   │   │   │   │   ├── color_00001.jpg
    │   │   │   │   │   ├── color_00002.jpg
    │   │   │   │   │   └── ...
    │   │   │   │   └── depth/
    │   │   │   └── 02/ ...
    │   │   └── 02/ ...
    │   └── phrases/
    │       ├── 01/ ...
    │       └── ...
    └── F02/ ...

What this file does:
    1. Scans the MIRACL folder tree to find every utterance (color image folder).
    2. For each utterance, reads the JPEG frames, runs dlib mouth detection,
       crops to 46×140 px, pads/samples to exactly 75 frames.
    3. Generates a synthetic .align-style label from the folder's word/phrase ID.
    4. Exposes build_miracl_dataset() which returns a tf.data.Dataset in
       exactly the same format as the GRID pipeline — drop-in compatible.

Usage:
    from data_loader_miracl import build_miracl_dataset
    miracl_train, miracl_test = build_miracl_dataset()

    # Then combine with GRID data:
    combined = grid_train.concatenate(miracl_train)
"""

import os
import glob
import random
from typing import List, Tuple

import cv2
import dlib
import numpy as np
import tensorflow as tf

from config import DLIB_MODEL_PATH, TARGET_FRAMES, TARGET_H, TARGET_W
from utils import char_to_num

# ─────────────────────────────────────────────────────────────────────────────
# MIRACL-VC1 label tables
# ─────────────────────────────────────────────────────────────────────────────

# Map word/phrase ID (as string "01"–"10") to the spoken text
MIRACL_WORDS = {
    "01": "begin",          # "Hegin" is a typo in the original paper — audio is "begin"
    "02": "choose",
    "03": "connection",
    "04": "navigation",
    "05": "next",
    "06": "previous",
    "07": "start",
    "08": "stop",
    "09": "hello",
    "10": "well",
}

MIRACL_PHRASES = {
    "01": "stop navigation",
    "02": "excuse me",
    "03": "i am sorry",
    "04": "thank you",
    "05": "good bye",
    "06": "i love this game",
    "07": "nice to meet you",
    "08": "you are welcome",
    "09": "how are you",
    "10": "have a good time",
}

# ─────────────────────────────────────────────────────────────────────────────
# Dlib setup (loaded once, reused across all calls)
# ─────────────────────────────────────────────────────────────────────────────

_detector  = None
_predictor = None
MOUTH_POINTS = list(range(48, 68))


def _get_dlib():
    """Lazily load dlib tools once and cache them."""
    global _detector, _predictor
    if _detector is None:
        if not os.path.exists(DLIB_MODEL_PATH):
            raise FileNotFoundError(
                f"dlib model not found at: {DLIB_MODEL_PATH}\n"
                "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        _detector  = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
    return _detector, _predictor


# ─────────────────────────────────────────────────────────────────────────────
# Core per-utterance loader
# ─────────────────────────────────────────────────────────────────────────────

def load_miracl_frames(instance_folder: str) -> np.ndarray:
    """
    Load one MIRACL utterance from its color image folder.

    Reads all JPEG frames in order, runs dlib mouth detection on each,
    crops to 46×140 grayscale, then pads or sub-samples to exactly 75 frames.

    Args:
        instance_folder: Path to a folder like  .../F01/words/01/01/

    Returns:
        np.ndarray of shape (75, 46, 140, 1) dtype float32, z-normalized.
    """
    detector, predictor = _get_dlib()

    # Read JPEG frames in sorted order (color_00001.jpg, color_00002.jpg, ...)
    jpg_paths = sorted(glob.glob(os.path.join(instance_folder, "*.jpg")))
    if not jpg_paths:
        # Some releases use .png
        jpg_paths = sorted(glob.glob(os.path.join(instance_folder, "*.png")))
    if not jpg_paths:
        raise ValueError(f"No image files found in: {instance_folder}")

    raw_crops = []

    for jpg_path in jpg_paths:
        frame = cv2.imread(jpg_path)
        if frame is None:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            # Fallback: blank frame — face not detected
            crop = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        else:
            face      = faces[0]
            landmarks = predictor(gray, face)
            coords    = np.array([
                [landmarks.part(n).x, landmarks.part(n).y]
                for n in MOUTH_POINTS
            ])
            x_min = max(int(coords[:, 0].min()) - 10, 0)
            x_max = min(int(coords[:, 0].max()) + 10, frame.shape[1])
            y_min = max(int(coords[:, 1].min()) - 10, 0)
            y_max = min(int(coords[:, 1].max()) + 10, frame.shape[0])
            mouth = gray[y_min:y_max, x_min:x_max]
            crop  = cv2.resize(mouth, (TARGET_W, TARGET_H))

        raw_crops.append(crop)

    if len(raw_crops) == 0:
        raise ValueError(f"Could not read any frames from: {instnace_folder}")

    # ── Normalise frame count to exactly TARGET_FRAMES (75) ──────────────────
    total = len(raw_crops)
    if total < TARGET_FRAMES:
        pad       = [raw_crops[-1]] * (TARGET_FRAMES - total)
        raw_crops = raw_crops + pad
    else:
        indices   = np.linspace(0, total - 1, TARGET_FRAMES, dtype=int)
        raw_crops = [raw_crops[i] for i in indices]

    # ── Stack → (75, 46, 140, 1) → z-normalize ───────────────────────────────
    arr  = np.stack(raw_crops, axis=0).astype(np.float32)   # (75, 46, 140)
    arr  = np.expand_dims(arr, axis=-1)                     # (75, 46, 140, 1)
    mean = arr.mean()
    std  = arr.std()
    if std > 0:
        arr = (arr - mean) / std
    else:
        arr = arr - mean

    return arr


def make_miracl_label(label_text: str) -> tf.Tensor:
    """
    Convert a spoken text string to an integer-encoded character label tensor.

    This matches exactly what load_alignments() produces for GRID data, so
    both datasets can be batched together.

    Args:
        label_text: e.g. "stop navigation" or "hello"

    Returns:
        tf.Tensor of shape (N_chars,) dtype int64 — padded to 40 in the batch.
    """
    chars  = list(label_text)   # split string into individual characters
    return char_to_num(
        tf.reshape(
            tf.strings.unicode_split(chars, input_encoding="UTF-8"),
            (-1,)
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def _collect_utterances(miracl_root: str, use_phrases: bool = True) -> List[Tuple[str, str]]:
    """
    Walk the MIRACL folder tree and return a list of (color_folder, label_text).

    Args:
        miracl_root:  Root of the downloaded dataset, e.g. "miraclvc1/"
        use_phrases:  If True, include phrase utterances. If False, words only.

    Returns:
        List of (path_to_color_folder, spoken_text) tuples.
    """
    utterances = []

    for speaker in sorted(os.listdir(miracl_root)):
        speaker_path = os.path.join(miracl_root, speaker)
        if not os.path.isdir(speaker_path):
            continue

        # ── Words ────────────────────────────────────────────────────────────
        words_path = os.path.join(speaker_path, "words")
        if os.path.isdir(words_path):
            for word_id in sorted(os.listdir(words_path)):
                label = MIRACL_WORDS.get(word_id)
                if label is None:
                    continue
                word_path = os.path.join(words_path, word_id)
                for instance in sorted(os.listdir(word_path)):
                    # The instance folder itself contains the JPEGs (no "color" subfolder)
                    instance_folder = os.path.join(word_path, instance)
                    # Check that there is at least one .jpg file
                    if glob.glob(os.path.join(instance_folder, "*.jpg")):
                        utterances.append((instance_folder, label))

        # ── Phrases ──────────────────────────────────────────────────────────
        if use_phrases:
            phrases_path = os.path.join(speaker_path, "phrases")
            if os.path.isdir(phrases_path):
                for phrase_id in sorted(os.listdir(phrases_path)):
                    label = MIRACL_PHRASES.get(phrase_id)
                    if label is None:
                        continue
                    phrase_path = os.path.join(phrases_path, phrase_id)
                    for instance in sorted(os.listdir(phrase_path)):
                        # The instance folder itself contains the JPEGs (no "color" subfolder)
                        instance_folder = os.path.join(phrase_path, instance)
                        # Check that there is at least one .jpg file
                        if glob.glob(os.path.join(instance_folder, "*.jpg")):
                            utterances.append((instance_folder, label))

    return utterances


def _load_one(color_folder_bytes, label_bytes):
    """
    tf.py_function wrapper — loads one utterance inside the tf.data pipeline.
    """
    color_folder = color_folder_bytes.numpy().decode("utf-8")
    label_text   = label_bytes.numpy().decode("utf-8")

    frames = load_miracl_frames(color_folder)
    label  = make_miracl_label(label_text)

    return frames, label


def _mappable(color_folder_tensor, label_tensor):
    """Wraps _load_one in tf.py_function for use in dataset.map()."""
    result = tf.py_function(
        _load_one,
        [color_folder_tensor, label_tensor],
        (tf.float32, tf.int64),
    )
    return result


def build_miracl_dataset(
    miracl_root: str,
    use_phrases: bool = True,
    train_split: float = 0.8,
    batch_size: int = 2,
    shuffle_seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build train and test tf.data.Datasets from the MIRACL-VC1 dataset.

    The output format is IDENTICAL to the GRID pipeline (padded_batch of
    (frames, labels)), so you can concatenate or interleave them directly.

    Args:
        miracl_root:   Root directory of the MIRACL dataset.
        use_phrases:   Whether to include phrase utterances (default True).
        train_split:   Fraction of data for training (default 0.8).
        batch_size:    Videos per batch (default 2, same as GRID).
        shuffle_seed:  Random seed for reproducibility.

    Returns:
        (train_dataset, test_dataset) — two tf.data.Datasets.

    Example:
        from data_loader_miracl import build_miracl_dataset

        miracl_train, miracl_test = build_miracl_dataset("miraclvc1/")

        # Combine with GRID:
        from dataset import train as grid_train, test as grid_test
        combined_train = grid_train.concatenate(miracl_train)
        combined_test  = grid_test.concatenate(miracl_test)
    """
    utterances = _collect_utterances(miracl_root, use_phrases=use_phrases)

    if len(utterances) == 0:
        raise ValueError(
            f"No utterances found in: {miracl_root}\n"
            "Check that the folder structure matches the expected layout."
        )

    print(f"Found {len(utterances)} MIRACL utterances total.")

    # Shuffle before split (deterministic with seed)
    random.seed(shuffle_seed)
    random.shuffle(utterances)

    split_idx = int(len(utterances) * train_split)
    train_utts = utterances[:split_idx]
    test_utts  = utterances[split_idx:]

    print(f"  Train: {len(train_utts)}  |  Test: {len(test_utts)}")

    def _make_ds(utts):
        folders = [u[0] for u in utts]
        labels  = [u[1] for u in utts]

        ds = tf.data.Dataset.from_tensor_slices((folders, labels))
        ds = ds.shuffle(len(utts), reshuffle_each_iteration=False)
        ds = ds.map(_mappable, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=(
                [TARGET_FRAMES, None, None, None],
                [40],
            ),
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return _make_ds(train_utts), _make_ds(test_utts)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ROOT = sys.argv[1] if len(sys.argv) > 1 else "miraclvc1"

    print(f"Scanning: {ROOT}")
    utterances = _collect_utterances(ROOT, use_phrases=True)
    print(f"Total utterances found: {len(utterances)}")

    if utterances:
        folder, label = utterances[0]
        print(f"\nTesting first utterance:")
        print(f"  Folder : {folder}")
        print(f"  Label  : {label}")
        frames = load_miracl_frames(folder)
        print(f"  Frames shape: {frames.shape}")
        print(f"  Mean: {frames.mean():.4f}  Std: {frames.std():.4f}")
