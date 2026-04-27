"""
config.py
---------
Single source of truth for every path and constant.

TARGET_H and TARGET_W are now REMOVED.
The model accepts variable spatial dimensions — crops are taken at native
resolution and the model uses GlobalAveragePooling2D to collapse them.

New crop-control parameters:
  CROP_PADDING_RATIO  — how much whitespace to add around the lip bounding box
                        as a fraction of the box size (0.5 = 50% padding each side)
  MIN_CROP_H / _W     — floor on crop size so very-low-res videos still have
                        enough pixels for the CNN kernels to operate on
  CROP_ALIGN_MULTIPLE — spatial dims are rounded up to this multiple so
                        MaxPool3D(1,2,2) divisions never produce fractional sizes
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "s1")
ALIGN_DIR       = os.path.join(BASE_DIR, "data", "alignments", "s1")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint")
DLIB_MODEL_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# ── Temporal dimension (fixed — CTC requires fixed time steps per batch) ──────

TARGET_FRAMES = 75    # 3 s × 25 fps

# ── GRID corpus fixed coarse crop (used in data_loader.py stage-1 ROI) ───────
# These slice the lower-face region before dlib runs, speeding up detection
# and reducing false positives on the fixed-camera GRID setup.
# They are NOT the final crop size — dlib refines the exact lip region within
# this ROI. The final crop dimensions vary per video (adaptive resolution).
GRID_CROP_Y = slice(190, 236)   # vertical ROI: rows 190–236
GRID_CROP_X = slice(80, 220)    # horizontal ROI: cols 80–220

# ── Spatial crop parameters (replaces fixed TARGET_H / TARGET_W) ──────────────

# Fraction of the lip bounding box width/height added as padding on ALL sides.
# 0.5 → crop is 2× the tight lip box in each dimension, giving natural context.
CROP_PADDING_RATIO = 0.5

# Minimum crop dimensions in pixels. Prevents the CNN from receiving patches
# smaller than its kernel sizes (3×5×5 / 3×3×3).
# Minimum crop dimensions in pixels.
#
# The architecture halves spatial dims 6 times total:
#   3x Conv3D(stride 1,2,2)  -> div2 each = div8
#   3x MaxPool3D(1,2,2)      -> div2 each = div8
#   Total reduction factor   = div64
#
# A crop of H pixels produces H/64 pixels before GlobalAveragePooling.
# If H < 64 the final MaxPool receives a spatial dim < 1 -> CRASH.
# Setting both to 64 guarantees at least a 1x1 feature map entering GAP.
MIN_CROP_H = 64
MIN_CROP_W = 64

# Must match the total spatial reduction factor (64).
# Crops are rounded UP to multiples of 64 so every dimension divides
# evenly through all 6 halving stages without producing fractional sizes.
CROP_ALIGN_MULTIPLE = 64

# ── Color / normalization (paper Appendix A.2) ────────────────────────────────

TARGET_C       = 3    # RGB
CHANNEL_MEANS  = [0.7136, 0.4906, 0.3283]   # [R, G, B] over GRID training set
CHANNEL_STDS   = [0.1138, 0.1078, 0.0917]

# ── Vocabulary ────────────────────────────────────────────────────────────────

VOCAB = list("abcdefghijklmnopqrstuvwxyz'?!123456789 ")

# ── Dataset ───────────────────────────────────────────────────────────────────

TRAIN_SIZE = 450
BATCH_SIZE = 2

# ── Training (paper Appendix A.1) ─────────────────────────────────────────────

LEARNING_RATE  = 0.0001
EPOCHS         = 100
LR_DECAY_EPOCH = 30

# ── CTC beam search (paper uses 200; 100 is CPU-practical) ────────────────────

BEAM_WIDTH = 100
