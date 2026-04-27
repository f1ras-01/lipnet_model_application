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
  - F01-F06 (words):   load from cropped/ — no dlib, faster and more accurate
  - F01-F10 (phrases): load from dataset/ with dlib (cropped/ has no phrases)
  - F07-F10 (words):   load from dataset/ with dlib
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

import glob
import random
import threading
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


# ── Dlib — thread-safe lazy initialization ────────────────────────────────────
#
# tf.data with AUTOTUNE runs map() in multiple parallel threads.
# Without a lock, Thread A can set _detector but not yet _predictor,
# while Thread B reads _detector (not None) and skips the init block,
# returning (detector, None) — causing "NoneType is not callable".
#
# Double-checked locking pattern: cheap check outside the lock,
# guaranteed-correct re-check inside the lock.

MOUTH_POINTS = list(range(48, 68))
_dlib_lock   = threading.Lock()
_detector    = None
_predictor   = None


def _get_dlib():
    """Return (detector, predictor), initializing them once in a thread-safe way."""
    global _detector, _predictor
    if _detector is None or _predictor is None:          # fast path (no lock needed usually)
        with _dlib_lock:
            if _detector is None or _predictor is None:  # re-check under lock
                if not os.path.exists(DLIB_MODEL_PATH):
                    raise FileNotFoundError(
                        f"dlib model not found: {DLIB_MODEL_PATH}\n"
                        "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                    )
                _detector  = dlib.get_frontal_face_detector()
                _predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
    return _detector, _predictor


# ─────────────────────────────────────────────────────────────────────────────
# Image reading helper
# ─────────────────────────────────────────────────────────────────────────────

def _read_color_images(instance_folder: str) -> List[np.ndarray]:
    """
    Read all color_*.jpg files from an instance folder in sorted order.
    Depth images (depth_*.png) are ignored.
    """
    paths = sorted(glob.glob(os.path.join(instance_folder, "color_*.jpg")))
    if not paths:
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
# Path A — Pre-cropped images (cropped/ folder, F01-F06 words only)
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_cropped(instance_folder: str, augment: bool = False) -> np.ndarray:
    """
    Load from the pre-cropped archive. No dlib needed — images are already
    mouth crops.

    Returns:
        np.ndarray (75, H, W, 3) float32.
    """
    raw = _read_color_images(instance_folder)

    if not raw:
        dummy = np.zeros((MIN_CROP_H, MIN_CROP_W, TARGET_C), dtype=np.uint8)
        raw   = [dummy] * TARGET_FRAMES

    if augment:
        raw = _frame_jitter(raw, p=0.05)
        if random.random() < 0.5:
            raw = [np.fliplr(f) for f in raw]

    total = len(raw)
    if total < TARGET_FRAMES:
        raw = raw + [raw[-1]] * (TARGET_FRAMES - total)
    else:
        idx = np.linspace(0, total - 1, TARGET_FRAMES, dtype=int)
        raw = [raw[i] for i in idx]

    size = canonical_size(raw)
    arr  = resize_to_canonical(raw, size)
    return normalize_rgb(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Path B — Full frames (dataset/ folder, uses dlib)
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_dataset(instance_folder: str, augment: bool = False) -> np.ndarray:
    """
    Load from full frames and crop the mouth via dlib.
    Used for speakers F07-F10 and all phrase utterances.

    Returns:
        np.ndarray (75, H, W, 3) float32.
    """
    # Get dlib tools — thread-safe, guaranteed both are non-None here
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

    # Forward fill then backward fill for frames with no detection
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

    if augment:
        raw_crops = _frame_jitter(raw_crops, p=0.05)
        if random.random() < 0.5:
            raw_crops = [np.fliplr(c) for c in raw_crops]

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
    Walk archive/ and return (instance_folder, label_text, source) tuples.
    source is "cropped" or "dataset" — controls which loader is used.
    Priority: use cropped/ when available, dataset/ as fallback.
    """
    dataset_root = os.path.join(archive_root, "dataset")
    cropped_root = os.path.join(archive_root, "cropped")

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"dataset/ subfolder not found in: {archive_root}\n"
            f"Expected: {dataset_root}"
        )

    utterances = []
    speakers   = sorted([
        s for s in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, s))
    ])

    for speaker in speakers:
        sp_cropped = os.path.join(cropped_root, speaker) if os.path.isdir(cropped_root) else None

        # ── Words ─────────────────────────────────────────────────────────────
        words_ds = os.path.join(dataset_root, speaker, "words")
        words_cr = os.path.join(sp_cropped, "words") if sp_cropped else None

        if os.path.isdir(words_ds):
            for word_id in sorted(os.listdir(words_ds)):
                label = MIRACL_WORDS.get(word_id)
                if label is None:
                    continue

                word_path_ds = os.path.join(words_ds, word_id)
                word_path_cr = os.path.join(words_cr, word_id) if (
                    words_cr and os.path.isdir(words_cr)
                ) else None

                for instance in sorted(os.listdir(word_path_ds)):
                    inst_cr = os.path.join(word_path_cr, instance) if word_path_cr else None
                    inst_ds = os.path.join(word_path_ds, instance)

                    if inst_cr and os.path.isdir(inst_cr):
                        utterances.append((inst_cr, label, "cropped"))
                    elif os.path.isdir(inst_ds):
                        utterances.append((inst_ds, label, "dataset"))

        # ── Phrases (always from dataset/ — cropped/ has no phrases) ──────────
        if use_phrases:
            phrases_ds = os.path.join(dataset_root, speaker, "phrases")
            if os.path.isdir(phrases_ds):
                for phrase_id in sorted(os.listdir(phrases_ds)):
                    label = MIRACL_PHRASES.get(phrase_id)
                    if label is None:
                        continue
                    phrase_path = os.path.join(phrases_ds, phrase_id)
                    for instance in sorted(os.listdir(phrase_path)):
                        inst_path = os.path.join(phrase_path, instance)
                        if os.path.isdir(inst_path):
                            utterances.append((inst_path, label, "dataset"))

    return utterances


# ─────────────────────────────────────────────────────────────────────────────
# Label encoder
# ─────────────────────────────────────────────────────────────────────────────

def make_miracl_label(label_text: str) -> tf.Tensor:
    """
    Convert spoken text to integer-encoded character tensor.
    Uses tf.constant on a flat list — GPU-safe, no RaggedTensor involved.
    """
    chars = list(label_text)   # plain Python list of single-char strings
    return char_to_num(
        tf.constant(chars, dtype=tf.string)
    )


# ─────────────────────────────────────────────────────────────────────────────
# tf.py_function wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _load_one(folder_bytes, label_bytes, source_bytes, augment: bool = False):
    """Route to pre-cropped or dlib path based on source field."""
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

    Automatically uses pre-cropped images for F01-F06 words and falls back
    to dlib detection for F07-F10 and all phrases.

    Output format is identical to the GRID pipeline — directly concatenable.

    Args:
        archive_root:  Path to the archive/ root folder.
        use_phrases:   Include phrase utterances (default True).
        train_split:   Fraction for training (default 0.8).
        batch_size:    Videos per batch (default 2).
        augment:       Apply H-flip + frame jitter to training split.
        shuffle_seed:  Seed for reproducibility.

    Returns:
        (train_dataset, test_dataset)
    """
    utterances = _collect_utterances(archive_root, use_phrases=use_phrases)

    if not utterances:
        raise ValueError(f"No utterances found in: {archive_root}")

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

    utts  = _collect_utterances(ROOT, use_phrases=True)
    n_cr  = sum(1 for u in utts if u[2] == "cropped")
    n_ds  = sum(1 for u in utts if u[2] == "dataset")
    print(f"Total utterances : {len(utts)}")
    print(f"  Pre-cropped    : {n_cr}  (no dlib)")
    print(f"  Full frames    : {n_ds}  (dlib required)")

    if utts:
        cr_utts = [u for u in utts if u[2] == "cropped"]
        if cr_utts:
            folder, label, _ = cr_utts[0]
            print(f"\n[Test] Pre-cropped: {folder}  label={label}")
            f = _load_from_cropped(folder)
            print(f"  Shape: {f.shape}  min={f.min():.3f}  max={f.max():.3f}")

        ds_utts = [u for u in utts if u[2] == "dataset"]
        if ds_utts:
            folder, label, _ = ds_utts[0]
            print(f"\n[Test] Full-frame (dlib): {folder}  label={label}")
            f = _load_from_dataset(folder)
            print(f"  Shape: {f.shape}  min={f.min():.3f}  max={f.max():.3f}")
