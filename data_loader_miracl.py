"""
data_loader_miracl.py
---------------------
Loader for the MIRACL dataset (archive/ folder structure).

Archive layout:
    archive/
    ├── dataset/          <- full frames, all speakers F01-F10
    │   ├── F01/
    │   │   ├── words/
    │   │   │   ├── 01/           <- word ID
    │   │   │   │   ├── 01/       <- instance number
    │   │   │   │   │   ├── color_001.jpg
    │   │   │   │   │   ├── color_002.jpg
    │   │   │   │   │   ├── depth_001.png   <- ignored
    │   │   │   │   │   └── ...
    │   │   │   │   └── 02/ ...
    │   │   │   └── 02/ ...
    │   │   └── phrases/
    │   │       └── (same layout)
    │   └── F02/ ... F10/
    └── cropped/          <- pre-cropped mouth images, speakers F01-F06 only
        ├── F01/
        │   ├── words/
        │   │   ├── 01/
        │   │   │   ├── 01/
        │   │   │   │   ├── color_001.jpg   <- already a mouth crop, no dlib needed
        │   │   │   │   └── ...
        │   │   │   └── 02/ ...
        │   │   └── 02/ ...
        │   └── (no phrases/ in cropped)
        └── F02/ ... F06/

Loading strategy:
  - F01–F06 (words):   load from cropped/ — no dlib, faster and more accurate
  - F01–F10 (phrases): load from dataset/ with dlib (cropped/ has no phrases)
  - F07–F10 (words):   load from dataset/ with dlib

The pre-cropped images are already mouth-region crops so they are passed
directly to canonical_size() + resize_to_canonical() without any detection step.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

import os
import glob
import random
from typing import List, Tuple

import cv2
import dlib
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from config import DLIB_MODEL_PATH, MIN_CROP_H, MIN_CROP_W, TARGET_C, TARGET_FRAMES
from crop_utils import canonical_size, crop_mouth, normalize_rgb, resize_to_canonical
from data_loader import _frame_jitter
from utils import char_to_num


# ── Label tables ──────────────────────────────────────────────────────────────

MIRACL_WORDS = {
    "01": "begin",      "02": "choose",     "03": "connection",
    "04": "navigation", "05": "next",       "06": "previous",
    "07": "start",      "08": "stop",       "09": "hello",
    "10": "well",
}

MIRACL_PHRASES = {
    "01": "stop navigation",  "02": "excuse me",
    "03": "i am sorry",       "04": "thank you",
    "05": "good bye",         "06": "i love this game",
    "07": "nice to meet you", "08": "you are welcome",
    "09": "how are you",      "10": "have a good time",
}

# ── Dlib (lazy-loaded once, only used when cropped/ is not available) ─────────

_detector  = None
_predictor = None
MOUTH_POINTS = list(range(48, 68))


def _get_dlib():
    global _detector, _predictor
    if _detector is None:
        if not os.path.exists(DLIB_MODEL_PATH):
            raise FileNotFoundError(
                f"dlib model not found: {DLIB_MODEL_PATH}\n"
                "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        _detector  = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
    return _detector, _predictor


# ─────────────────────────────────────────────────────────────────────────────
# Image reading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_color_images(instance_folder: str) -> List[np.ndarray]:
    """
    Read all color_*.jpg files from an instance folder in sorted order.
    Depth images (depth_*.png) are ignored.

    Args:
        instance_folder: Path like .../F01/words/01/01/

    Returns:
        List of RGB np.ndarray images (uint8, variable size).
        Returns empty list if no images found.
    """
    # Glob only color images — explicitly exclude depth
    paths = sorted(glob.glob(os.path.join(instance_folder, "color_*.jpg")))
    if not paths:
        # Fallback: any jpg that isn't depth
        paths = sorted([
            p for p in glob.glob(os.path.join(instance_folder, "*.jpg"))
            if "depth" not in os.path.basename(p).lower()
        ])

    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


# ─────────────────────────────────────────────────────────────────────────────
# Path A — Pre-cropped images (cropped/ folder, F01–F06 words only)
# No dlib needed — images are already mouth crops
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_cropped(instance_folder: str, augment: bool = False) -> np.ndarray:
    """
    Load an utterance from the pre-cropped archive.

    Images are already mouth crops — just read, optionally augment,
    standardise frame count, and normalise.

    Args:
        instance_folder: Path like archive/cropped/F01/words/01/01/
        augment:         Apply H-flip + frame jitter if True.

    Returns:
        np.ndarray (75, H, W, 3) float32.
    """
    raw = _read_color_images(instance_folder)

    if not raw:
        dummy = np.zeros((MIN_CROP_H, MIN_CROP_W, TARGET_C), dtype=np.uint8)
        raw   = [dummy] * TARGET_FRAMES

    # Augmentation
    if augment:
        raw = _frame_jitter(raw, p=0.05)
        if random.random() < 0.5:
            raw = [np.fliplr(f) for f in raw]

    # Fix frame count to exactly TARGET_FRAMES
    total = len(raw)
    if total < TARGET_FRAMES:
        raw = raw + [raw[-1]] * (TARGET_FRAMES - total)
    else:
        idx = np.linspace(0, total - 1, TARGET_FRAMES, dtype=int)
        raw = [raw[i] for i in idx]

    # Standardise spatial dims within this utterance + normalise
    size = canonical_size(raw)
    arr  = resize_to_canonical(raw, size)   # (75, H, W, 3) uint8
    return normalize_rgb(arr)               # (75, H, W, 3) float32


# ─────────────────────────────────────────────────────────────────────────────
# Path B — Full frames (dataset/ folder, all speakers, all categories)
# Uses dlib to detect and crop the mouth region
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_dataset(instance_folder: str, augment: bool = False) -> np.ndarray:
    """
    Load an utterance from full frames and crop the mouth via dlib.

    Used for:
      - Speakers F07–F10 (not present in cropped/)
      - All phrase utterances (cropped/ has no phrases subfolder)

    Args:
        instance_folder: Path like archive/dataset/F07/words/01/01/
        augment:         Apply H-flip + frame jitter if True.

    Returns:
        np.ndarray (75, H, W, 3) float32.
    """
    detector, predictor = _get_dlib()
    raw_frames = _read_color_images(instance_folder)

    if not raw_frames:
        dummy = np.zeros((MIN_CROP_H, MIN_CROP_W, TARGET_C), dtype=np.uint8)
        return normalize_rgb(
            resize_to_canonical([dummy] * TARGET_FRAMES, (MIN_CROP_H, MIN_CROP_W))
        )

    raw_crops = []

    for frame_rgb in raw_frames:
        gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)

        if not faces:
            raw_crops.append(None)
        else:
            face      = faces[0]
            landmarks = predictor(gray, face)
            raw_crops.append(crop_mouth(frame_rgb, landmarks))

    # Forward + backward fill for frames with no detection
    last = None
    for i in range(len(raw_crops)):
        if raw_crops[i] is not None:
            last = raw_crops[i]
        elif last is not None:
            raw_crops[i] = last.copy()
    last = None
    for i in range(len(raw_crops) - 1, -1, -1):
        if raw_crops[i] is not None:
            last = raw_crops[i]
        elif last is not None:
            raw_crops[i] = last.copy()

    dummy = np.zeros((MIN_CROP_H, MIN_CROP_W, TARGET_C), dtype=np.uint8)
    raw_crops = [c if c is not None else dummy for c in raw_crops]

    # Augmentation
    if augment:
        raw_crops = _frame_jitter(raw_crops, p=0.05)
        if random.random() < 0.5:
            raw_crops = [np.fliplr(c) for c in raw_crops]

    # Fix frame count
    total = len(raw_crops)
    if total < TARGET_FRAMES:
        raw_crops = raw_crops + [raw_crops[-1]] * (TARGET_FRAMES - total)
    else:
        idx       = np.linspace(0, total - 1, TARGET_FRAMES, dtype=int)
        raw_crops = [raw_crops[i] for i in idx]

    size = canonical_size(raw_crops)
    arr  = resize_to_canonical(raw_crops, size)
    return normalize_rgb(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Utterance collection
# ─────────────────────────────────────────────────────────────────────────────

def _collect_utterances(
    archive_root: str,
    use_phrases: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    Walk the archive folder tree and return all utterances.

    Returns a list of (instance_folder, label_text, source) tuples where
    source is either "cropped" or "dataset", used by the loader to pick the
    right loading path (dlib vs pre-cropped).

    Priority: use cropped/ when available, dataset/ as fallback.

    Args:
        archive_root: Path to the archive/ folder.
        use_phrases:  Include phrase utterances (from dataset/ only).

    Returns:
        List of (instance_folder, label_text, source) tuples.
    """
    dataset_root = os.path.join(archive_root, "dataset")
    cropped_root = os.path.join(archive_root, "cropped")

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"dataset/ subfolder not found in: {archive_root}\n"
            f"Expected: {dataset_root}"
        )

    utterances = []

    # Collect all speakers from dataset/ (authoritative list)
    speakers = sorted([
        s for s in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, s))
    ])

    for speaker in speakers:
        # Determine which categories are pre-cropped for this speaker
        speaker_cropped = os.path.join(cropped_root, speaker) if os.path.isdir(cropped_root) else None

        # ── Words ─────────────────────────────────────────────────────────────
        words_dataset = os.path.join(dataset_root, speaker, "words")
        words_cropped = os.path.join(speaker_cropped, "words") if speaker_cropped else None

        if os.path.isdir(words_dataset):
            for word_id in sorted(os.listdir(words_dataset)):
                label = MIRACL_WORDS.get(word_id)
                if label is None:
                    continue

                word_path_ds = os.path.join(words_dataset, word_id)
                word_path_cr = os.path.join(words_cropped, word_id) if (
                    words_cropped and os.path.isdir(words_cropped)
                ) else None

                for instance in sorted(os.listdir(word_path_ds)):
                    inst_cropped = os.path.join(word_path_cr, instance) if word_path_cr else None
                    inst_dataset = os.path.join(word_path_ds, instance)

                    # Use pre-cropped if available for this instance
                    if inst_cropped and os.path.isdir(inst_cropped):
                        utterances.append((inst_cropped, label, "cropped"))
                    elif os.path.isdir(inst_dataset):
                        utterances.append((inst_dataset, label, "dataset"))

        # ── Phrases ───────────────────────────────────────────────────────────
        # cropped/ has no phrases — always load from dataset/ with dlib
        if use_phrases:
            phrases_dataset = os.path.join(dataset_root, speaker, "phrases")
            if os.path.isdir(phrases_dataset):
                for phrase_id in sorted(os.listdir(phrases_dataset)):
                    label = MIRACL_PHRASES.get(phrase_id)
                    if label is None:
                        continue
                    phrase_path = os.path.join(phrases_dataset, phrase_id)
                    for instance in sorted(os.listdir(phrase_path)):
                        inst_path = os.path.join(phrase_path, instance)
                        if os.path.isdir(inst_path):
                            utterances.append((inst_path, label, "dataset"))

    return utterances


# ─────────────────────────────────────────────────────────────────────────────
# Label encoder
# ─────────────────────────────────────────────────────────────────────────────

def make_miracl_label(label_text: str) -> tf.Tensor:
    """Convert spoken text string to integer-encoded character tensor."""
    chars = list(label_text)
    return char_to_num(
        tf.reshape(tf.strings.unicode_split(chars, input_encoding="UTF-8"), (-1,))
    )


# ─────────────────────────────────────────────────────────────────────────────
# tf.py_function wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _load_one(folder_bytes, label_bytes, source_bytes, augment: bool = False):
    """Load one utterance — routes to pre-cropped or dlib path automatically."""
    folder = folder_bytes.numpy().decode("utf-8")
    label  = label_bytes.numpy().decode("utf-8")
    source = source_bytes.numpy().decode("utf-8")

    if source == "cropped":
        frames = _load_from_cropped(folder, augment=augment)
    else:
        frames = _load_from_dataset(folder, augment=augment)

    return frames, make_miracl_label(label)


# ─────────────────────────────────────────────────────────────────────────────
# Public dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_miracl_dataset(
    archive_root: str,
    use_phrases:  bool  = True,
    train_split:  float = 0.8,
    batch_size:   int   = 2,
    augment:      bool  = False,
    shuffle_seed: int   = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build train and test tf.data.Datasets from the MIRACL archive folder.

    Automatically uses pre-cropped images for speakers F01–F06 (words) and
    falls back to dlib detection from full frames for F07–F10 and all phrases.

    Output format is identical to the GRID pipeline — directly concatenable
    with dataset.train / dataset.test.

    Args:
        archive_root:  Path to the archive/ root folder.
        use_phrases:   Include phrase utterances (default True).
        train_split:   Fraction for training (default 0.8).
        batch_size:    Videos per batch (default 2).
        augment:       Apply H-flip + frame jitter to training set.
        shuffle_seed:  Seed for reproducibility.

    Returns:
        (train_dataset, test_dataset)

    Example:
        from data_loader_miracl import build_miracl_dataset

        miracl_train, miracl_test = build_miracl_dataset("archive/")

        # Combine with GRID:
        from dataset import train as grid_train, test as grid_test
        combined = grid_train.concatenate(miracl_train)
    """
    utterances = _collect_utterances(archive_root, use_phrases=use_phrases)

    if not utterances:
        raise ValueError(f"No utterances found in: {archive_root}")

    # Count breakdown for user info
    n_cropped = sum(1 for u in utterances if u[2] == "cropped")
    n_dataset = sum(1 for u in utterances if u[2] == "dataset")
    print(f"MIRACL archive: {len(utterances)} utterances total")
    print(f"  Pre-cropped (no dlib): {n_cropped}")
    print(f"  Full frames (dlib):    {n_dataset}")

    random.seed(shuffle_seed)
    random.shuffle(utterances)

    split   = int(len(utterances) * train_split)
    tr_utts = utterances[:split]
    te_utts = utterances[split:]
    print(f"  Train: {len(tr_utts)}  |  Test: {len(te_utts)}")

    # padded_shapes: None,None for H,W — variable resolution
    _shapes = ([TARGET_FRAMES, None, None, None], [40])

    def _make_ds(utts, do_aug):
        folders = [u[0] for u in utts]
        labels  = [u[1] for u in utts]
        sources = [u[2] for u in utts]

        ds = tf.data.Dataset.from_tensor_slices((folders, labels, sources))
        ds = ds.shuffle(len(utts), reshuffle_each_iteration=False)
        ds = ds.map(
            lambda f, l, s: tf.py_function(
                lambda fb, lb, sb: _load_one(fb, lb, sb, augment=do_aug),
                [f, l, s],
                (tf.float32, tf.int64),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.padded_batch(batch_size, padded_shapes=_shapes)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return _make_ds(tr_utts, do_aug=augment), _make_ds(te_utts, do_aug=False)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ROOT = sys.argv[1] if len(sys.argv) > 1 else "archive"

    print(f"Scanning: {ROOT}\n")
    utts = _collect_utterances(ROOT, use_phrases=True)

    n_cr = sum(1 for u in utts if u[2] == "cropped")
    n_ds = sum(1 for u in utts if u[2] == "dataset")
    print(f"Total utterances : {len(utts)}")
    print(f"  Pre-cropped    : {n_cr}  (no dlib)")
    print(f"  Full frames    : {n_ds}  (dlib required)")

    if utts:
        # Test one cropped utterance
        cropped_utts = [u for u in utts if u[2] == "cropped"]
        if cropped_utts:
            folder, label, source = cropped_utts[0]
            print(f"\n[Test] Pre-cropped utterance")
            print(f"  Folder : {folder}")
            print(f"  Label  : {label}")
            frames = _load_from_cropped(folder)
            print(f"  Shape  : {frames.shape}  min={frames.min():.3f}  max={frames.max():.3f}")

        # Test one dataset utterance
        dataset_utts = [u for u in utts if u[2] == "dataset"]
        if dataset_utts:
            folder, label, source = dataset_utts[0]
            print(f"\n[Test] Full-frame utterance (dlib)")
            print(f"  Folder : {folder}")
            print(f"  Label  : {label}")
            frames = _load_from_dataset(folder)
            print(f"  Shape  : {frames.shape}  min={frames.min():.3f}  max={frames.max():.3f}")
