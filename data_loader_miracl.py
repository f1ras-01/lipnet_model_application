"""
data_loader_miracl.py
---------------------
MIRACL-VC1 adapter — updated for adaptive spatial resolution.

Each utterance is cropped at native resolution via dlib, producing variable
H×W tensors that are compatible with the same padded_batch pipeline as GRID.
"""

import os
import glob
import random
from typing import List, Tuple

import cv2
import dlib
import numpy as np
import tensorflow as tf

from config import DLIB_MODEL_PATH, MIN_CROP_H, MIN_CROP_W, TARGET_C, TARGET_FRAMES
from crop_utils import canonical_size, crop_mouth, normalize_rgb, resize_to_canonical
from data_loader import _frame_jitter
from utils import char_to_num

MIRACL_WORDS = {
    "01": "begin",     "02": "choose",     "03": "connection", "04": "navigation",
    "05": "next",      "06": "previous",   "07": "start",      "08": "stop",
    "09": "hello",     "10": "well",
}
MIRACL_PHRASES = {
    "01": "stop navigation",  "02": "excuse me",       "03": "i am sorry",
    "04": "thank you",        "05": "good bye",         "06": "i love this game",
    "07": "nice to meet you", "08": "you are welcome",  "09": "how are you",
    "10": "have a good time",
}

_detector  = None
_predictor = None
MOUTH_POINTS = list(range(48, 68))


def _get_dlib():
    global _detector, _predictor
    if _detector is None:
        if not os.path.exists(DLIB_MODEL_PATH):
            raise FileNotFoundError(f"dlib model not found: {DLIB_MODEL_PATH}")
        _detector  = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
    return _detector, _predictor


def load_miracl_frames(color_folder: str, augment: bool = False) -> np.ndarray:
    """
    Load one MIRACL utterance from its color image folder.

    Returns variable-resolution RGB frames normalized with GRID statistics,
    making the output format identical to load_video() output.

    Args:
        color_folder: Path like .../F01/words/01/01/color/
        augment:      Apply H-flip and frame jitter if True.

    Returns:
        np.ndarray (75, H, W, 3) float32.  H, W are multiples of 8.
    """
    detector, predictor = _get_dlib()

    jpg_paths = sorted(glob.glob(os.path.join(color_folder, "*.jpg")))
    if not jpg_paths:
        jpg_paths = sorted(glob.glob(os.path.join(color_folder, "*.png")))
    if not jpg_paths:
        raise ValueError(f"No images in: {color_folder}")

    raw_crops = []

    for jpg_path in jpg_paths:
        frame = cv2.imread(jpg_path)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces     = detector(gray)

        if len(faces) == 0:
            raw_crops.append(None)
        else:
            face      = faces[0]
            landmarks = predictor(gray, face)
            raw_crops.append(crop_mouth(frame_rgb, landmarks))

    if not raw_crops:
        raise ValueError(f"No readable frames in: {color_folder}")

    # Fill None placeholders (forward + backward fill)
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


def make_miracl_label(label_text: str) -> tf.Tensor:
    chars = list(label_text)
    return char_to_num(
        tf.reshape(tf.strings.unicode_split(chars, input_encoding="UTF-8"), (-1,))
    )


def _collect_utterances(root: str, use_phrases: bool = True) -> List[Tuple[str, str]]:
    utterances = []
    for speaker in sorted(os.listdir(root)):
        sp = os.path.join(root, speaker)
        if not os.path.isdir(sp):
            continue
        for cat, lmap in [("words", MIRACL_WORDS), ("phrases", MIRACL_PHRASES)]:
            if cat == "phrases" and not use_phrases:
                continue
            cat_path = os.path.join(sp, cat)
            if not os.path.isdir(cat_path):
                continue
            for lid in sorted(os.listdir(cat_path)):
                label = lmap.get(lid)
                if label is None:
                    continue
                for inst in sorted(os.listdir(os.path.join(cat_path, lid))):
                    cf = os.path.join(cat_path, lid, inst, "color")
                    if os.path.isdir(cf):
                        utterances.append((cf, label))
    return utterances


def build_miracl_dataset(
    miracl_root: str,
    use_phrases: bool = True,
    train_split: float = 0.8,
    batch_size:  int   = 2,
    augment:     bool  = False,
    shuffle_seed: int  = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build MIRACL train/test datasets with variable spatial dimensions.
    Output format is identical to the GRID pipeline — directly concatenable.
    """
    utterances = _collect_utterances(miracl_root, use_phrases)
    if not utterances:
        raise ValueError(f"No utterances found in: {miracl_root}")

    print(f"MIRACL: {len(utterances)} utterances.")
    random.seed(shuffle_seed)
    random.shuffle(utterances)

    split   = int(len(utterances) * train_split)
    tr_utts = utterances[:split]
    te_utts = utterances[split:]
    print(f"  Train {len(tr_utts)}  |  Test {len(te_utts)}")

    _shapes = ([TARGET_FRAMES, None, None, None], [40])

    def _make_ds(utts, do_aug):
        folders = [u[0] for u in utts]
        labels  = [u[1] for u in utts]
        ds = tf.data.Dataset.from_tensor_slices((folders, labels))
        ds = ds.shuffle(len(utts), reshuffle_each_iteration=False)
        ds = ds.map(
            lambda f, l: tf.py_function(
                lambda fb, lb: (
                    load_miracl_frames(fb.numpy().decode(), augment=do_aug),
                    make_miracl_label(lb.numpy().decode()),
                ),
                [f, l], (tf.float32, tf.int64),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.padded_batch(batch_size, padded_shapes=_shapes)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return _make_ds(tr_utts, do_aug=augment), _make_ds(te_utts, do_aug=False)
