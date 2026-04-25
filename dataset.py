"""
dataset.py
----------
Builds the tf.data pipeline used for training and validation.

Exports:
    train   — batched, shuffled, prefetched dataset (450 batches)
    test    — remaining batches used for validation and the example callback

Pipeline steps:
    1. list_files     — glob all .mpg files in DATA_DIR
    2. shuffle        — randomise order once (reshuffle=False for reproducibility)
    3. map            — run load_data on each file path via mappable_function
    4. padded_batch   — group into batches of 2, padding alignments to length 40
    5. prefetch       — overlap GPU compute with CPU data loading
    6. take / skip    — train/test split
"""

import tensorflow as tf

from config import BATCH_SIZE, DATA_DIR, TARGET_FRAMES, TRAIN_SIZE
from data_loader import mappable_function

# ── Build the full dataset ────────────────────────────────────────────────────

data = tf.data.Dataset.list_files(f"{DATA_DIR}/*.mpg")
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(
    BATCH_SIZE,
    padded_shapes=(
        [TARGET_FRAMES, None, None, None],  # frames — pad spatial dims if needed
        [40],                               # alignments — pad to max label length
    )
)
data = data.prefetch(tf.data.AUTOTUNE)

# ── Train / test split ────────────────────────────────────────────────────────

train = data.take(TRAIN_SIZE)
test  = data.skip(TRAIN_SIZE)

# ─────────────────────────────────────────────────────────────────────────────
# Quick inspection (run this file directly to verify the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Total batches : {len(data)}")
    print(f"Train batches : {len(train)}")
    print(f"Test batches  : {len(test)}")
    print()

    sample = data.as_numpy_iterator().next()
    frames_batch, align_batch = sample

    print(f"frames_batch shape : {frames_batch.shape}")  # (2, 75, 46, 140, 1)
    print(f"align_batch shape  : {align_batch.shape}")   # (2, 40)
