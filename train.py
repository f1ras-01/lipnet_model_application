"""
train.py
--------
Trains the LipNet model from scratch (or resumes from a checkpoint).

Run with:
    python train.py

What this script does:
    1. Configures GPU memory growth so TF doesn't grab all VRAM at once.
    2. Builds the dataset pipeline (train + test splits).
    3. Builds the model.
    4. Compiles with Adam + CTCLoss.
    5. Attaches three callbacks: checkpoint saver, LR scheduler, example printer.
    6. Calls model.fit() for EPOCHS epochs.

After training, weights are saved to:
    models/checkpoint
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ── GPU setup ─────────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU detected: {physical_devices[0]}")
except (IndexError, RuntimeError):
    print("No GPU found — training on CPU (this will be slow).")

# ── Project imports ───────────────────────────────────────────────────────────
from config import CHECKPOINT_PATH, EPOCHS, LEARNING_RATE, MODEL_DIR
from dataset import test, train
from model import CTCLoss, build_model, get_callbacks

# ── Build and compile ─────────────────────────────────────────────────────────
model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=CTCLoss,
)
model.summary()

# ── (Optional) Resume from existing checkpoint ───────────────────────────────
if os.path.exists(CHECKPOINT_PATH + ".index"):
    print(f"\nResuming from checkpoint: {CHECKPOINT_PATH}")
    model.load_weights(CHECKPOINT_PATH)
else:
    print("\nNo checkpoint found — starting from scratch.")

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
callbacks = get_callbacks(test)

print(f"\nTraining for {EPOCHS} epochs …")
model.fit(
    train,
    validation_data=test,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print(f"\nTraining complete. Weights saved to: {CHECKPOINT_PATH}")
