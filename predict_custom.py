"""
predict_custom.py
-----------------
Run the trained model on any recorded video.

The model now accepts ANY resolution — the crop size adapts to the face
size in YOUR video, and the model handles it natively via GAP.

Run with:
    python predict_custom.py --video "path/to/video.mp4"
    python predict_custom.py --video "path/to/video.mp4" --beam 200
    python predict_custom.py --video "path/to/video.mp4" --greedy
"""

import argparse
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except (IndexError, RuntimeError):
    pass

from config import BEAM_WIDTH, CHECKPOINT_PATH
from data_loader import load_custom_video
from model import CTCLoss, build_model, ctc_beam_decode
from utils import num_to_char

parser = argparse.ArgumentParser()
parser.add_argument("--video",  default=None)
parser.add_argument("--beam",   type=int, default=BEAM_WIDTH)
parser.add_argument("--greedy", action="store_true")
args = parser.parse_args()

VIDEO_PATH = args.video or r"C:\path\to\your\video.mp4"

print("Loading model ...")
model = build_model()
model.compile(optimizer="adam", loss=CTCLoss)
model.load_weights(CHECKPOINT_PATH).expect_partial()
print(f"Weights loaded: {CHECKPOINT_PATH}\n")

print(f"Processing: {VIDEO_PATH}")
sample = load_custom_video(VIDEO_PATH)
print(f"  Final tensor shape: {sample.shape}")   # (75, H, W, 3) — H,W adapt to video
print()

input_data = np.expand_dims(sample, axis=0)      # (1, 75, H, W, 3)
print("Running model.predict ...")
yhat = model.predict(input_data)

if args.greedy:
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    method  = "greedy"
else:
    decoded = ctc_beam_decode(yhat, beam_width=args.beam)
    method  = f"beam search (width={args.beam})"

prediction = tf.strings.reduce_join(
    [num_to_char(w) for w in decoded[0]]
).numpy().decode("utf-8")

print(f"\n{'~'*80}")
print(f"PREDICTION ({method}): {prediction}")
print(f"{'~'*80}\n")
