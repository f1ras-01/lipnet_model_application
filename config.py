"""
config.py
---------
Single source of truth for every path and constant used across the project.
Change values here and every other file picks them up automatically.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────

# Root of the project (folder that contains this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GRID corpus data
DATA_DIR        = os.path.join(BASE_DIR, "data", "s1")          # .mpg files
ALIGN_DIR       = os.path.join(BASE_DIR, "data", "alignments", "s1")  # .align files

# Saved model weights
MODEL_DIR       = os.path.join(BASE_DIR, "models")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint")

# dlib landmark model — must be downloaded manually and placed here
DLIB_MODEL_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# ── Model input dimensions ────────────────────────────────────────────────────

TARGET_FRAMES = 75      # Number of frames per video clip
TARGET_H      = 46      # Frame height in pixels  (mouth crop)
TARGET_W      = 140     # Frame width in pixels   (mouth crop)

# Pixel crop for the GRID corpus videos (fixed camera, fixed speaker position)
# These values slice [y_start:y_end, x_start:x_end] from each 360×288 grayscale frame
GRID_CROP_Y = slice(190, 236)   # height → 46 px
GRID_CROP_X = slice(80, 220)    # width  → 140 px

# ── Vocabulary ────────────────────────────────────────────────────────────────

VOCAB = list("abcdefghijklmnopqrstuvwxyz'?!123456789 ")

# ── Dataset split ─────────────────────────────────────────────────────────────

TRAIN_SIZE  = 450       # Number of batches kept for training
BATCH_SIZE  = 2         # Videos per batch

# ── Training hyperparameters ──────────────────────────────────────────────────

LEARNING_RATE   = 0.0001
EPOCHS          = 100
LR_DECAY_EPOCH  = 30    # Epoch after which learning rate starts decaying
LR_DECAY_FACTOR = 0.1   # tf.math.exp(-0.1) ≈ 0.905 per epoch
