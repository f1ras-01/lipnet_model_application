"""
predict_custom.py
-----------------
Run the trained model on YOUR OWN recorded video files.

This is the script you use every time you record a new video of yourself and
want to test it. You only need to change VIDEO_PATH — everything else is
automatic.

Run with:
    python predict_custom.py
    python predict_custom.py --video "path/to/your/video.mp4"

What happens internally:
    1. dlib detects your face and mouth in every frame.
    2. The mouth region is cropped and resized to 46×140 px (grayscale).
    3. The clip is padded or sub-sampled to exactly 75 frames.
    4. The model runs inference and CTC decodes the output.
    5. The predicted text is printed.
"""

import argparse

import numpy as np
import tensorflow as tf

# ── GPU setup ─────────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except (IndexError, RuntimeError):
    pass

# ── Project imports ───────────────────────────────────────────────────────────
from config import CHECKPOINT_PATH
from data_loader import load_custom_video
from model import CTCLoss, build_model
from utils import num_to_char

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Predict lip-reading output on a custom video.")
parser.add_argument(
    "--video",
    required=False,
    default=None,
    help="Path to your recorded video (.mp4 / .mpg / .avi / .mov).",
)
args = parser.parse_args()

# ── Set your video path here (used when --video is not passed via CLI) ────────
VIDEO_PATH = args.video or r"C:\users\firas\onedrive\desktop\9rayaaaa\pfa\pfa2\files\RL_tests\vids\place red at g six again.mpg"
# ─────────────────────────────────────────────────────────────────────────────

# ── Build model and load trained weights ──────────────────────────────────────
print("Loading model …")
model = build_model()
model.compile(optimizer="adam", loss=CTCLoss)
model.load_weights(CHECKPOINT_PATH).expect_partial()
print(f"Weights loaded from: {CHECKPOINT_PATH}\n")

# ── Preprocess the custom video ───────────────────────────────────────────────
print(f"Processing video: {VIDEO_PATH}")
sample_fixed = load_custom_video(VIDEO_PATH)

print(f"  Preprocessed shape : {sample_fixed.shape}")   # should be (75, 46, 140, 1)
print(f"  Value range        : [{sample_fixed.min():.3f}, {sample_fixed.max():.3f}]")
print()

# ── Run inference ─────────────────────────────────────────────────────────────
input_data = np.expand_dims(sample_fixed, axis=0)   # (1, 75, 46, 140, 1)
print("Running model.predict …")
yhat = model.predict(input_data)

# ── Decode CTC output ─────────────────────────────────────────────────────────
decoded = tf.keras.backend.ctc_decode(
    yhat, input_length=[75], greedy=True
)[0][0].numpy()

prediction = tf.strings.reduce_join(
    [num_to_char(word) for word in decoded[0]]
).numpy().decode("utf-8")

print(f"\n{'~'*80}")
print(f"PREDICTION: {prediction}")
print(f"{'~'*80}\n")
