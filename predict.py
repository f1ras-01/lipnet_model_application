"""
predict.py
----------
Run the trained model on videos from the GRID corpus (data/s1/*.mpg).

Use this when you want to test prediction quality on the original dataset
that the model was trained on. It loads a real .mpg file, runs inference,
and prints the ground-truth label alongside the model prediction.

Run with:
    python predict.py
    python predict.py --video data/s1/swwv9a.mpg

Edit VIDEO_PATH below or pass --video on the command line.
"""

import argparse
import os

import tensorflow as tf
import numpy as np

# ── GPU setup ─────────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except (IndexError, RuntimeError):
    pass

# ── Project imports ───────────────────────────────────────────────────────────
from config import CHECKPOINT_PATH, DATA_DIR
from data_loader import load_data
from model import CTCLoss, build_model
from utils import num_to_char

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Predict on a GRID corpus video.")
parser.add_argument(
    "--video",
    default=os.path.join(DATA_DIR, "swwv9a.mpg"),
    help="Path to a GRID .mpg file (default: data/s1/swwv9a.mpg)",
)
args = parser.parse_args()

VIDEO_PATH = args.video

# ── Build model and load weights ──────────────────────────────────────────────
model = build_model()
model.compile(optimizer="adam", loss=CTCLoss)   # compile needed before load_weights
model.load_weights(CHECKPOINT_PATH)
print(f"Loaded weights from: {CHECKPOINT_PATH}\n")

# ── Load one GRID video ───────────────────────────────────────────────────────
print(f"Loading video: {VIDEO_PATH}")
frames, alignments = load_data(tf.constant(VIDEO_PATH))

# ── Ground truth ──────────────────────────────────────────────────────────────
real_text = tf.strings.reduce_join(
    [num_to_char(word) for word in alignments]
).numpy().decode("utf-8")
print(f"{'~'*80}")
print(f"REAL TEXT   : {real_text}")

# ── Run inference ─────────────────────────────────────────────────────────────
# model.predict expects a batch dimension → expand from (75,46,140,1) to (1,75,46,140,1)
input_batch = tf.expand_dims(frames, axis=0)
yhat        = model.predict(input_batch)

# ── Decode CTC output ─────────────────────────────────────────────────────────
decoded = tf.keras.backend.ctc_decode(
    yhat, input_length=[75], greedy=True
)[0][0].numpy()

prediction = tf.strings.reduce_join(
    [num_to_char(word) for word in decoded[0]]
).numpy().decode("utf-8")

print(f"PREDICTION  : {prediction}")
print(f"{'~'*80}")
