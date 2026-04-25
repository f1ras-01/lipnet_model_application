"""
dataset.py
----------
tf.data pipeline — updated for variable spatial dimensions.

The key change is in padded_batch:
  - frames padded_shape: [TARGET_FRAMES, None, None, None]
    The None, None for H and W tells TF to pad each batch to the MAXIMUM
    H and W seen in that batch. Videos with smaller crops get zero-padded
    on the right and bottom edges.
  - GlobalAveragePooling2D in the model averages over H×W, so the zero
    padding has a small diluting effect (≈proportional to how much padding
    was added) rather than corrupting the features.

Training: augmented (H-flip + frame jitter)
Test:     no augmentation
"""

import tensorflow as tf

from config import BATCH_SIZE, DATA_DIR, TARGET_FRAMES, TRAIN_SIZE
from data_loader import mappable_function, mappable_function_augment

# List all .mpg files and shuffle once
data_all = tf.data.Dataset.list_files(f"{DATA_DIR}/*.mpg")
data_all = data_all.shuffle(500, reshuffle_each_iteration=False)

train_files = data_all.take(TRAIN_SIZE)
test_files  = data_all.skip(TRAIN_SIZE)

# padded_shapes: None for H and W means "pad to batch maximum"
_padded_shapes = (
    [TARGET_FRAMES, None, None, None],  # frames: (T, H, W, C) — H, W vary
    [40],                               # labels: pad to max label length
)

train = (
    train_files
    .map(mappable_function_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(BATCH_SIZE, padded_shapes=_padded_shapes)
    .prefetch(tf.data.AUTOTUNE)
)

test = (
    test_files
    .map(mappable_function, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(BATCH_SIZE, padded_shapes=_padded_shapes)
    .prefetch(tf.data.AUTOTUNE)
)

if __name__ == "__main__":
    print(f"Train batches: {len(train)}")
    print(f"Test batches : {len(test)}")
    sample = train.as_numpy_iterator().next()
    print(f"frames shape : {sample[0].shape}")   # (2, 75, H, W, 3) — H,W = max in batch
    print(f"labels shape : {sample[1].shape}")
