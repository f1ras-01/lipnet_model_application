"""
model.py
--------
LipNet — fully variable-resolution architecture.

The model now accepts ANY spatial input size because:
  - Conv3D layers are fully convolutional (no fixed spatial dim dependency).
  - TimeDistributed(GlobalAveragePooling2D) collapses (H, W, C) -> C regardless
    of H and W. This replaces the old TimeDistributed(Flatten) which required
    a fixed spatial size.
  - Input shape declared as (TARGET_FRAMES, None, None, TARGET_C).

Effect on the pipeline:
  - A 360p video produces a ~48×64 mouth crop  -> GAP -> 96-dim feature
  - A 1080p video produces a ~120×160 mouth crop -> GAP -> 96-dim feature
  - Both produce the same (batch, 75, 96) tensor entering the Bi-GRU.
  - Training batches: padded_batch pads smaller crops in a batch to match
    the largest, then GAP collapses the padding away cleanly.

Architecture (paper-accurate for all other components):
  Input:   (B, 75, H, W, 3)       H and W vary per video
  Block 1: Conv3D(32, 3x5x5, stride 1,2,2) + ReLU + Dropout(0.5) + MaxPool3D(1,2,2)
  Block 2: Conv3D(64, 3x5x5, stride 1,2,2) + ReLU + Dropout(0.5) + MaxPool3D(1,2,2)
  Block 3: Conv3D(96, 3x3x3, stride 1,2,2) + ReLU + Dropout(0.5) + MaxPool3D(1,2,2)
  GAP:     TimeDistributed(GlobalAveragePooling2D)  -> (B, 75, 96)
  BiGRU1:  Bidirectional(GRU(256)) + return_sequences=True -> (B, 75, 512)
  BiGRU2:  Bidirectional(GRU(256)) + return_sequences=True -> (B, 75, 512)
  Output:  Dense(vocab+1, softmax)                          -> (B, 75, 41)
"""

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation, Bidirectional, Conv3D, Dense, Dropout,
    GlobalAveragePooling2D, GRU, MaxPool3D, TimeDistributed,
)
from tensorflow.keras.models import Sequential

from config import (
    BEAM_WIDTH, CHECKPOINT_PATH, EPOCHS, LEARNING_RATE,
    LR_DECAY_EPOCH, TARGET_C, TARGET_FRAMES,
)
from utils import VOCAB_SIZE


def build_model() -> Sequential:
    """
    Construct and return the variable-resolution LipNet model (uncompiled).

    The spatial input dimensions (H, W) are fully dynamic — pass None for both.
    The model compiles and runs identically regardless of crop size.
    """
    model = Sequential(name="LipNet_VarRes")

    # ── Block 1 ───────────────────────────────────────────────────────────────
    # input_shape: (frames, height, width, channels)
    # None, None for height and width = accepts any spatial size
    model.add(Conv3D(
        32,
        kernel_size=(3, 5, 5),
        strides=(1, 2, 2),          # spatial halving in the conv itself
        padding="same",
        kernel_initializer="he_normal",
        input_shape=(TARGET_FRAMES, None, None, TARGET_C),
    ))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))   # spatial /2 again

    # ── Block 2 ───────────────────────────────────────────────────────────────
    model.add(Conv3D(
        64,
        kernel_size=(3, 5, 5),
        strides=(1, 2, 2),
        padding="same",
        kernel_initializer="he_normal",
    ))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # ── Block 3 ───────────────────────────────────────────────────────────────
    model.add(Conv3D(
        96,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        padding="same",
        kernel_initializer="he_normal",
    ))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # ── Global Average Pooling — the key to variable spatial input ─────────────
    #
    # After 3x (stride-2 conv + MaxPool), spatial dims are divided by 2^6 = 64.
    # For a 64×96 crop  -> 1×1  after 6 halvings  (minimum, GAP gives shape-1 output)
    # For a 128×192 crop -> 2×3  -> GAP -> 96
    # For a 256×384 crop -> 4×6  -> GAP -> 96
    #
    # GlobalAveragePooling2D averages across ALL spatial positions → always 96-dim.
    # Padding added by padded_batch contributes all-zero features that are averaged
    # in, but the CNN features from real pixels dominate — the effect is minimal.
    model.add(TimeDistributed(GlobalAveragePooling2D()))   # (B, 75, 96)

    # ── Bi-GRU sequence encoder ───────────────────────────────────────────────
    model.add(Bidirectional(GRU(
        256,
        return_sequences=True,
        kernel_initializer="he_normal",
        recurrent_initializer="orthogonal",
    )))
    model.add(Bidirectional(GRU(
        256,
        return_sequences=True,
        kernel_initializer="he_normal",
        recurrent_initializer="orthogonal",
    )))

    # ── CTC output ────────────────────────────────────────────────────────────
    model.add(Dense(
        VOCAB_SIZE + 1,
        kernel_initializer="he_normal",
        activation="softmax",
    ))

    return model


# ── CTC loss ─────────────────────────────────────────────────────────────────

def CTCLoss(y_true, y_pred):
    batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# ── LR schedule ──────────────────────────────────────────────────────────────

def scheduler(epoch, lr):
    if epoch < LR_DECAY_EPOCH:
        return lr
    return float(lr * tf.math.exp(-0.1))


# ── CTC beam decode ───────────────────────────────────────────────────────────

def ctc_beam_decode(y_pred, input_length=None, beam_width=None):
    if beam_width is None:
        beam_width = BEAM_WIDTH
    if input_length is None:
        input_length = [y_pred.shape[1]] * y_pred.shape[0]
    decoded, _ = tf.keras.backend.ctc_decode(
        y_pred, input_length=input_length, greedy=False, beam_width=beam_width,
    )
    return decoded[0].numpy()


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(test_dataset):
    from utils import num_to_char

    class ProduceExample(tf.keras.callbacks.Callback):
        def __init__(self, dataset):
            self.dataset = dataset.as_numpy_iterator()

        def on_epoch_end(self, epoch, logs=None):
            data    = self.dataset.next()
            yhat    = self.model.predict(data[0])
            decoded = ctc_beam_decode(yhat)
            for i in range(len(yhat)):
                original   = tf.strings.reduce_join(num_to_char(data[1][i])).numpy().decode("utf-8")
                prediction = tf.strings.reduce_join(num_to_char(decoded[i])).numpy().decode("utf-8")
                print(f"Original  : {original}")
                print(f"Prediction: {prediction}")
                print("~" * 100)

    return [
        ModelCheckpoint(CHECKPOINT_PATH, monitor="loss", save_weights_only=True),
        LearningRateScheduler(scheduler),
        ProduceExample(test_dataset),
    ]


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"\nInput  : {model.input_shape}")
    print(f"Output : {model.output_shape}")

    # Verify with two different spatial sizes
    import numpy as np
    for h, w in [(48, 64), (96, 128), (192, 256)]:
        dummy = np.zeros((1, TARGET_FRAMES, h, w, TARGET_C), dtype=np.float32)
        out   = model(dummy, training=False)
        print(f"  Input ({h}x{w}) -> Output {out.shape}  ✓")
