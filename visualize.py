"""
visualize.py
------------
Visualisation tools — updated to show adaptive crop regions.

visualize_landmarks(video_path)
    Full frame with face box, landmark dots, and the ADAPTIVE crop box
    (red dashed) sized by CROP_PADDING_RATIO — not a fixed pixel rectangle.

visualize_mouth_crops(video_path)
    The actual variable-size crops at native resolution that the pipeline
    would feed to the model. Each tile shows its own dimensions.

Run with:
    python visualize.py --video "path/to/video.mp4" --mode both
"""

import argparse
import os

import cv2
import dlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from config import DLIB_MODEL_PATH, MIN_CROP_H, MIN_CROP_W, TARGET_C
from crop_utils import canonical_size, crop_mouth, normalize_rgb


MOUTH_POINTS = list(range(48, 68))


def _get_dlib():
    if not os.path.exists(DLIB_MODEL_PATH):
        raise FileNotFoundError(f"dlib model not found: {DLIB_MODEL_PATH}")
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(DLIB_MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Function 1 — Full frame with adaptive crop box
# ─────────────────────────────────────────────────────────────────────────────

def visualize_landmarks(video_path: str, num_frames_to_show: int = 6) -> None:
    """
    Show sampled frames with landmarks and the ADAPTIVE crop rectangle.

    The crop box reflects the actual CROP_PADDING_RATIO — it grows with the
    face size, shrinks for distant faces, and is always aligned to 8 pixels.
    """
    from crop_utils import CROP_PADDING_RATIO
    detector, predictor = _get_dlib()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h_nat = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_nat = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Video: {os.path.basename(video_path)}  |  {w_nat}×{h_nat} px  |  {total} frames")

    indices = np.linspace(0, total - 1, num_frames_to_show, dtype=int)
    cols    = num_frames_to_show // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 5, 8))
    axes = axes.flatten()

    for plot_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ax        = axes[plot_idx]
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {frame_idx}", fontsize=9)
        ax.axis("off")

        faces = detector(gray)
        if not faces:
            ax.set_title(f"Frame {frame_idx} — NO FACE", color="red", fontsize=9)
            continue

        face      = faces[0]
        landmarks = predictor(gray, face)

        # Face box (green)
        ax.add_patch(patches.Rectangle(
            (face.left(), face.top()), face.width(), face.height(),
            linewidth=2, edgecolor="lime", facecolor="none",
        ))

        # All 68 landmarks (blue)
        all_c = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)])
        ax.scatter(all_c[:, 0], all_c[:, 1], s=6, c="dodgerblue", zorder=5)

        # Mouth landmarks (red)
        mc = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in MOUTH_POINTS])
        ax.scatter(mc[:, 0], mc[:, 1], s=16, c="red", zorder=6)

        # Adaptive crop box (red dashed)
        lip_w = mc[:, 0].max() - mc[:, 0].min()
        lip_h = mc[:, 1].max() - mc[:, 1].min()
        pad_x = lip_w * CROP_PADDING_RATIO
        pad_y = lip_h * CROP_PADDING_RATIO
        bx = max(mc[:, 0].min() - pad_x, 0)
        by = max(mc[:, 1].min() - pad_y, 0)
        bw = min(mc[:, 0].max() + pad_x, frame_rgb.shape[1]) - bx
        bh = min(mc[:, 1].max() + pad_y, frame_rgb.shape[0]) - by
        ax.add_patch(patches.Rectangle(
            (bx, by), bw, bh,
            linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
        ))
        ax.text(bx, by - 4, f"adaptive {int(bw)}×{int(bh)}px",
                color="red", fontsize=6, va="bottom")

    cap.release()
    plt.suptitle(
        f"Adaptive Crop Visualization — {os.path.basename(video_path)}\n"
        f"🟢 Face box  🔵 All landmarks  🔴 Adaptive lip crop  "
        f"(padding ratio: {CROP_PADDING_RATIO})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Function 2 — Adaptive mouth crops (model input)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_mouth_crops(video_path: str, num_frames_to_show: int = 15) -> None:
    """
    Show the actual variable-size crops that would enter the model.
    Each tile displays its own pixel dimensions.
    Also prints what canonical_size() would choose for batch standardisation.
    """
    detector, predictor = _get_dlib()

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames_to_show, dtype=int)

    crops, labels = [], []

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces     = detector(gray)

        if not faces:
            crop = np.zeros((MIN_CROP_H, MIN_CROP_W, TARGET_C), dtype=np.uint8)
            labels.append(f"F{frame_idx}\n(no face)")
        else:
            face = faces[0]
            lm   = predictor(gray, face)
            crop = crop_mouth(frame_rgb, lm)
            labels.append(f"Frame {frame_idx}")

        crops.append(crop)

    cap.release()

    # Show canonical size
    size = canonical_size(crops)
    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"Canonical batch size (median): {size[1]}×{size[0]} px")
    print(f"Showing {len(crops)} frames  —  each at its own native crop size\n")

    cols = 5
    rows = (len(crops) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.2))
    axes = axes.flatten()

    for i, (crop, label) in enumerate(zip(crops, labels)):
        axes[i].imshow(crop)
        axes[i].set_title(label, fontsize=7)
        axes[i].set_xlabel(f"{crop.shape[1]}×{crop.shape[0]}px", fontsize=6)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for j in range(len(crops), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Adaptive Mouth Crops — {os.path.basename(video_path)}\n"
        f"Each tile = native resolution crop  |  Canonical batch size: {size[1]}×{size[0]} px",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default=None)
    parser.add_argument("--mode",   choices=["landmarks", "crops", "both"], default="both")
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args()

    VIDEO_PATH = args.video or r"C:\path\to\your\video.mp4"

    if args.mode in ("landmarks", "both"):
        visualize_landmarks(VIDEO_PATH, num_frames_to_show=args.frames)
    if args.mode in ("crops", "both"):
        visualize_mouth_crops(VIDEO_PATH, num_frames_to_show=15)
