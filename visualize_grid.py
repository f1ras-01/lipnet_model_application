"""
visualize_grid.py
-----------------
Diagnostic visualisation tools for GRID corpus videos (data/s1/*.mpg).

No face detection is used here. The GRID corpus has a fixed camera with the
speaker always in the same position, so the mouth region is simply sliced out
with the hardcoded pixel crop [190:236, 80:220] — exactly the same crop used
during training in load_video().

Three functions:

  visualize_raw_frames(video_path)
      Shows sampled raw colour frames straight from the .mpg file with the
      mouth crop region drawn as a red rectangle, so you can visually confirm
      the crop is landing on the right area.

  visualize_grid_crops(video_path)
      Shows the actual 46×140 grayscale crops that load_video() produces —
      i.e. the exact pixel values (before z-normalization) the model sees.

  visualize_normalized_crops(video_path)
      Shows the same crops after z-normalization — i.e. the exact tensor
      values that are passed to model.predict().

Run with:
    python visualize_grid.py
    python visualize_grid.py --video data/s1/swwv9a.mpg --mode all
    python visualize_grid.py --video data/s1/bbal6n.mpg --mode raw
    python visualize_grid.py --video data/s1/swwv9a.mpg --mode crops
    python visualize_grid.py --video data/s1/bbal6n.mpg --mode normalized
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf

from config import (
    DATA_DIR,
    ALIGN_DIR,
    GRID_CROP_X,
    GRID_CROP_Y,
    TARGET_H,
    TARGET_W,
    TARGET_FRAMES,
)
from utils import num_to_char
from data_loader import load_video, load_alignments


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _read_raw_frames(video_path: str) -> list:
    """Read all raw BGR frames from a video file and return as a list."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from: {video_path}")

    return frames


def _get_alignment_text(video_path: str) -> str:
    """
    Try to load the alignment text for a video.
    Returns the label string, or '(no alignment file found)' if missing.
    """
    file_name      = os.path.splitext(os.path.basename(video_path))[0]
    alignment_path = os.path.join(ALIGN_DIR, f"{file_name}.align")

    if not os.path.exists(alignment_path):
        return "(no alignment file found)"

    try:
        encoded = load_alignments(alignment_path)
        text    = tf.strings.reduce_join(
            [num_to_char(c) for c in encoded]
        ).numpy().decode("utf-8")
        return text
    except Exception as e:
        return f"(error reading alignment: {e})"


# ─────────────────────────────────────────────────────────────────────────────
# Function 1 — Raw colour frames with crop box overlay
# ─────────────────────────────────────────────────────────────────────────────

def visualize_raw_frames(video_path: str, num_frames_to_show: int = 6) -> None:
    """
    Show sampled raw colour frames from the .mpg file.

    A red dashed rectangle marks the [190:236, 80:220] crop region so you
    can confirm the fixed crop lands correctly on the speaker's mouth.

    Args:
        video_path:         Path to a GRID .mpg file.
        num_frames_to_show: Number of frames to sample (must be even).
    """
    raw_frames = _read_raw_frames(video_path)
    total      = len(raw_frames)
    label      = _get_alignment_text(video_path)

    print(f"Video      : {os.path.basename(video_path)}")
    print(f"Label      : {label}")
    print(f"Total frames: {total}  |  Frame size: {raw_frames[0].shape[1]}×{raw_frames[0].shape[0]} px")

    indices = np.linspace(0, total - 1, num_frames_to_show, dtype=int)

    cols = num_frames_to_show // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 5, 8))
    axes = axes.flatten()

    # Crop boundaries from config
    y_start = GRID_CROP_Y.start   # 190
    y_stop  = GRID_CROP_Y.stop    # 236
    x_start = GRID_CROP_X.start   # 80
    x_stop  = GRID_CROP_X.stop    # 220

    for plot_idx, frame_idx in enumerate(indices):
        frame_bgr = raw_frames[frame_idx]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        ax = axes[plot_idx]
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {frame_idx}", fontsize=10)
        ax.axis("off")

        # Draw crop rectangle
        crop_rect = patches.Rectangle(
            (x_start, y_start),
            x_stop - x_start,    # width  = 140
            y_stop  - y_start,   # height = 46
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(crop_rect)

        # Label the rectangle
        ax.text(
            x_start, y_start - 4,
            f"crop [{y_start}:{y_stop}, {x_start}:{x_stop}]",
            color="red", fontsize=7, va="bottom",
        )

    plt.suptitle(
        f"Raw Frames — {os.path.basename(video_path)}\n"
        f'Label: "{label}"\n'
        f"Red dashed box = mouth crop region used for training",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Function 2 — Raw grayscale mouth crops (before normalization)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_grid_crops(video_path: str, num_frames_to_show: int = 15) -> None:
    """
    Show the raw 46×140 grayscale mouth crops, before z-normalization.

    These are the pixel values straight from the fixed crop, converted to
    grayscale — the intermediate step before the mean/std normalization that
    load_video() applies.

    Args:
        video_path:         Path to a GRID .mpg file.
        num_frames_to_show: Number of frames to sample and display.
    """
    raw_frames = _read_raw_frames(video_path)
    total      = len(raw_frames)
    label      = _get_alignment_text(video_path)

    indices = np.linspace(0, total - 1, num_frames_to_show, dtype=int)

    crops  = []
    labels = []

    for frame_idx in indices:
        frame_bgr  = raw_frames[frame_idx]
        gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)         # (H, W)
        crop       = gray[GRID_CROP_Y, GRID_CROP_X]                      # (46, 140)
        crops.append(crop)
        labels.append(f"Frame {frame_idx}")

    # Plot
    cols = 5
    rows = (len(crops) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = axes.flatten()

    for i, (crop, lbl) in enumerate(zip(crops, labels)):
        axes[i].imshow(crop, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(lbl, fontsize=8)
        axes[i].set_xlabel(f"{crop.shape[1]}×{crop.shape[0]}px  |  "
                           f"min={crop.min()}  max={crop.max()}", fontsize=6)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for j in range(len(crops), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Raw Grayscale Crops (before normalization) — {os.path.basename(video_path)}\n"
        f'Label: "{label}"  |  Crop: [{GRID_CROP_Y.start}:{GRID_CROP_Y.stop}, '
        f'{GRID_CROP_X.start}:{GRID_CROP_X.stop}]  →  {TARGET_W}×{TARGET_H} px',
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    print(f"\n✅ {len(crops)} frames shown out of {total} total.")
    print(f"   Crop shape: {TARGET_W}×{TARGET_H} px  |  Pixel range: [0, 255]")
    print(f"   Label: {label}")


# ─────────────────────────────────────────────────────────────────────────────
# Function 3 — Z-normalized crops (exactly what the model receives)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_normalized_crops(video_path: str, num_frames_to_show: int = 15) -> None:
    """
    Show the z-normalized frames exactly as the model receives them.

    Calls load_video() directly so the output is guaranteed to be the
    identical tensor the model sees during both training and inference.
    Useful for confirming that custom video normalization matches this.

    Args:
        video_path:         Path to a GRID .mpg file.
        num_frames_to_show: Number of frames to sample and display.
    """
    label = _get_alignment_text(video_path)

    # Use the real load_video() — same function used during training
    frames_tensor = load_video(video_path)          # (N, 46, 140, 1)  float32
    frames_np     = frames_tensor.numpy()           # convert to numpy for plotting
    total         = frames_np.shape[0]

    print(f"\nload_video() output:")
    print(f"  Shape : {frames_np.shape}")
    print(f"  dtype : {frames_np.dtype}")
    print(f"  mean  : {frames_np.mean():.4f}  (should be ≈ 0.0)")
    print(f"  std   : {frames_np.std():.4f}   (should be ≈ 1.0)")
    print(f"  min   : {frames_np.min():.4f}")
    print(f"  max   : {frames_np.max():.4f}")
    print(f"  Label : {label}\n")

    indices = np.linspace(0, total - 1, num_frames_to_show, dtype=int)
    crops   = [frames_np[i, :, :, 0] for i in indices]   # drop channel dim for imshow
    labels  = [f"Frame {i}" for i in indices]

    # Use a diverging colormap so negative values are blue and positive are red,
    # making it easy to see the z-normalized distribution
    vmax = max(abs(frames_np.min()), abs(frames_np.max()))

    cols = 5
    rows = (len(crops) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = axes.flatten()

    for i, (crop, lbl) in enumerate(zip(crops, labels)):
        im = axes[i].imshow(crop, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i].set_title(lbl, fontsize=8)
        axes[i].set_xlabel(f"min={crop.min():.2f}  max={crop.max():.2f}", fontsize=6)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for j in range(len(crops), len(axes)):
        axes[j].axis("off")

    # Shared colorbar
    fig.colorbar(im, ax=axes[:len(crops)], orientation="vertical",
                 fraction=0.02, pad=0.02, label="z-score")

    plt.suptitle(
        f"Z-Normalized Crops (model input) — {os.path.basename(video_path)}\n"
        f'Label: "{label}"  |  Blue = negative  Red = positive  |  '
        f"mean≈{frames_np.mean():.3f}  std≈{frames_np.std():.3f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise GRID corpus video frames (no face detection)."
    )
    parser.add_argument(
        "--video",
        required=False,
        default=None,
        help="Path to a GRID .mpg file (default: first file found in DATA_DIR).",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "crops", "normalized", "all"],
        default="all",
        help=(
            "What to display:\n"
            "  raw        - colour frames with crop box overlay\n"
            "  crops      - raw grayscale mouth crops\n"
            "  normalized - z-normalized crops (exact model input)\n"
            "  all        - all three views  (default)"
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=6,
        help="Number of frames to sample in the raw view (default: 6, must be even).",
    )
    args = parser.parse_args()

    # ── Resolve video path ────────────────────────────────────────────────────
    if args.video:
        VIDEO_PATH = args.video
    else:
        # Default: pick the first .mpg in DATA_DIR
        import glob
        found = sorted(glob.glob(os.path.join(DATA_DIR, "*.mpg")))
        if not found:
            raise FileNotFoundError(
                f"No .mpg files found in {DATA_DIR}.\n"
                "Pass a path explicitly with --video."
            )
        VIDEO_PATH = found[0]
        print(f"No --video given. Using: {VIDEO_PATH}\n")

    # ── Run selected views ────────────────────────────────────────────────────
    if args.mode in ("raw", "all"):
        visualize_raw_frames(VIDEO_PATH, num_frames_to_show=args.frames)

    if args.mode in ("crops", "all"):
        visualize_grid_crops(VIDEO_PATH, num_frames_to_show=15)

    if args.mode in ("normalized", "all"):
        visualize_normalized_crops(VIDEO_PATH, num_frames_to_show=15)
