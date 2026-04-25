"""
model.py
--------
Defines and returns the LipNet model architecture.

Architecture (as in the original Nick Nochnack implementation):

    Input: (batch, 75 frames, 46 px, 140 px, 1 channel)
    │
    ├─ Conv3D(128, 3×3×3) + ReLU + MaxPool3D(1,2,2)    → (75, 23, 70, 128)
    ├─ Conv3D(256, 3×3×3) + ReLU + MaxPool3D(1,2,2)    → (75, 11, 35, 256)
    ├─ Conv3D(75,  3×3×3) + ReLU + MaxPool3D(1,2,2)    → (75,  5, 17,  75)
    │
    ├─ TimeDistributed(Flatten)                         → (75, 6375)
    │
    ├─ BiLSTM(128, return_sequences=True) + Dropout(0.5) → (75, 256)
    ├─ BiLSTM(128, return_sequences=True) + Dropout(0.5) → (75, 256)
    │
    └─ Dense(vocab_size+1, softmax)                     → (75, 41)

The +1 in the Dense output is for the CTC blank token.

Exports:
    build_model()  — constructs and returns an uncompiled Sequential model.
"""

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation, Bidirectional, Conv3D, Dense, Dropout,
    Flatten, LSTM, MaxPool3D, TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import CHECKPOINT_PATH, EPOCHS, LEARNING_RATE, LR_DECAY_EPOCH, TARGET_FRAMES, TARGET_H, TARGET_W
from utils import VOCAB_SIZE


def build_model() -> Sequential:
    """
    Construct the LipNet Sequential model and return it (uncompiled).

    Call model.compile() and model.load_weights() separately depending on
    whether you are training or just doing inference.
    """
    model = Sequential(name="LipNet")

    # ── Spatiotemporal convolutional front-end ──────────────────────────────
    # Block 1
    model.add(Conv3D(128, 3, input_shape=(TARGET_FRAMES, TARGET_H, TARGET_W, 1), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    # Block 2
    model.add(Conv3D(256, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    # Block 3
    model.add(Conv3D(75, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPool3D((1, 2, 2)))

    # ── Flatten spatial dims at each time-step ──────────────────────────────
    # After 3× (1,2,2) pooling: 46→5 (height), 140→17 (width), 75 filters
    # → 5 × 17 × 75 = 6375 features per time-step
    model.add(TimeDistributed(Flatten()))

    # ── Bidirectional LSTM sequence encoder ────────────────────────────────
    model.add(Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)))
    model.add(Dropout(0.5))

    # ── CTC output layer ────────────────────────────────────────────────────
    # +1 for the CTC blank token (index 0 in the output distribution)
    model.add(Dense(VOCAB_SIZE + 1, kernel_initializer="he_normal", activation="softmax"))

    return model


# ── Loss and scheduler (shared between train.py and here for convenience) ────

def CTCLoss(y_true, y_pred):
    """
    Connectionist Temporal Classification loss.

    Computes the CTC batch cost. Works with variable-length label sequences
    because the GRID alignment lengths vary per utterance.
    """
    import tensorflow as tf
    batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def scheduler(epoch, lr):
    """
    Learning-rate schedule:
    - First 30 epochs: keep lr constant.
    - After epoch 30: decay by factor exp(-0.1) ≈ 0.905 each epoch.
    """
    import tensorflow as tf
    if epoch < LR_DECAY_EPOCH:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def get_callbacks(test_dataset):
    """
    Returns the three standard training callbacks:
      - ModelCheckpoint   saves weights after every epoch if loss improved.
      - LearningRateScheduler  applies the decay schedule above.
      - ProduceExample    prints a side-by-side prediction vs. ground truth
                          at the end of every epoch.
    """
    import tensorflow as tf
    from utils import num_to_char

    class ProduceExample(tf.keras.callbacks.Callback):
        """Prints a prediction vs. ground truth pair at the end of each epoch."""

        def __init__(self, dataset):
            self.dataset = dataset.as_numpy_iterator()

        def on_epoch_end(self, epoch, logs=None):
            data  = self.dataset.next()
            yhat  = self.model.predict(data[0])
            decoded = tf.keras.backend.ctc_decode(yhat, [75, 75], greedy=False)[0][0].numpy()
            for i in range(len(yhat)):
                original   = tf.strings.reduce_join(num_to_char(data[1][i])).numpy().decode("utf-8")
                prediction = tf.strings.reduce_join(num_to_char(decoded[i])).numpy().decode("utf-8")
                print(f"Original  : {original}")
                print(f"Prediction: {prediction}")
                print("~" * 100)

    checkpoint_callback = ModelCheckpoint(
        CHECKPOINT_PATH, monitor="loss", save_weights_only=True
    )
    schedule_callback = LearningRateScheduler(scheduler)
    example_callback  = ProduceExample(test_dataset)

    return [checkpoint_callback, schedule_callback, example_callback]


# ─────────────────────────────────────────────────────────────────────────────
# Quick inspection (run this file directly to print the model summary)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"\nInput shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
