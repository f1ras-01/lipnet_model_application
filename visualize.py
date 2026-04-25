"""
visualize.py
------------
Diagnostic visualisation tools. Run this when you want to check whether
dlib is correctly detecting your face and cropping the mouth region.

Two functions:

  visualize_landmarks(video_path)
      Displays a grid of sampled frames with the 68 facial landmark dots,
      face bounding box, and the mouth crop bounding box drawn on top.

  visualize_mouth_crops(video_path)
      Displays the actual 46×140 grayscale images that would be fed into
      the model — exactly what the model sees.

Run with:
    python visualize.py
    python visualize.py --video "path/to/your/video.mp4" --mode both
    python visualize.py --video "path/to/your/video.mp4" --mode landmarks
    python visualize.py --video "path/to/your/video.mp4" --mode crops
"""

import argparse

import cv2
import dlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from config import DLIB_MODEL_PATH, TARGET_H, TARGET_W


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: load dlib tools once and reuse
# ─────────────────────────────────────────────────────────────────────────────

def _get_dlib_tools():
    """Return (detector, predictor, MOUTH_POINTS) — loads from disk once."""
    import os
    if not os.path.exists(DLIB_MODEL_PATH):
        raise FileNotFoundError(
            f"dlib model not found at: {DLIB_MODEL_PATH}\n"
            "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
    mouth_pts = list(range(48, 68))
    return detector, predictor, mouth_pts


# ─────────────────────────────────────────────────────────────────────────────
# Function 1: Landmarks on full frames
# ─────────────────────────────────────────────────────────────────────────────

def visualize_landmarks(video_path: str, num_frames_to_show: int = 6) -> None:
    """
    Show a grid of sampled video frames with facial landmarks overlaid.

    Each subplot shows:
      - Green rectangle  → face bounding box detected by dlib
      - Blue dots        → all 68 facial landmark points
      - Red dots         → only the 20 mouth landmarks (points 48–67)
      - Red dashed box   → exact crop region that goes into the model

    Args:
        video_path:        Path to any video file.
        num_frames_to_show: How many frames to sample and display (must be even).
    """
    detector, predictor, MOUTH_POINTS = _get_dlib_tools()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    # Evenly spaced frame indices
    sample_indices = np.linspace(0, total_frames - 1, num_frames_to_show, dtype=int)

    cols = num_frames_to_show // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 5, 8))
    axes = axes.flatten()

    for plot_idx, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ax = axes[plot_idx]
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {frame_idx}", fontsize=10)
        ax.axis("off")

        faces = detector(gray)

        if len(faces) == 0:
            ax.set_title(f"Frame {frame_idx} — NO FACE DETECTED", color="red", fontsize=9)
            continue

        face      = faces[0]
        landmarks = predictor(gray, face)

        # Green face bounding box
        face_rect = patches.Rectangle(
            (face.left(), face.top()), face.width(), face.height(),
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(face_rect)

        # Blue dots — all 68 landmarks
        all_coords = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)])
        ax.scatter(all_coords[:, 0], all_coords[:, 1], s=8, c="dodgerblue", zorder=5)

        # Red dots — mouth landmarks only
        mouth_coords = np.array([
            [landmarks.part(n).x, landmarks.part(n).y] for n in MOUTH_POINTS
        ])
        ax.scatter(mouth_coords[:, 0], mouth_coords[:, 1], s=20, c="red", zorder=6)

        # Red dashed box — mouth crop region
        x_min = mouth_coords[:, 0].min() - 10
        x_max = mouth_coords[:, 0].max() + 10
        y_min = mouth_coords[:, 1].min() - 10
        y_max = mouth_coords[:, 1].max() + 10
        mouth_rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
        )
        ax.add_patch(mouth_rect)

    cap.release()

    plt.suptitle(
        "Face Landmarks Visualization\n"
        "🟢 Face box   🔵 All 68 landmarks   🔴 Mouth landmarks & crop box",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Function 2: Cropped mouth frames (model's actual input)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_mouth_crops(
    video_path: str,
    num_frames_to_show: int = 15,
    target_h: int = TARGET_H,
    target_w: int = TARGET_W,
) -> None:
    """
    Show the cropped, resized mouth frames that would actually enter the model.

    Each tile is exactly target_w × target_h (140×46) grayscale pixels —
    identical to what load_custom_video() produces.

    Args:
        video_path:         Path to any video file.
        num_frames_to_show: Number of frames to sample and display.
        target_h:           Crop height (should match model input, default 46).
        target_w:           Crop width  (should match model input, default 140).
    """
    detector, predictor, MOUTH_POINTS = _get_dlib_tools()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, num_frames_to_show, dtype=int)

    crops  = []
    labels = []

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            mouth_resized = np.zeros((target_h, target_w), dtype=np.uint8)
            labels.append(f"F{frame_idx}\n(no face)")
        else:
            face      = faces[0]
            landmarks = predictor(gray, face)
            mouth_coords = np.array([
                [landmarks.part(n).x, landmarks.part(n).y] for n in MOUTH_POINTS
            ])
            x_min = max(int(mouth_coords[:, 0].min()) - 10, 0)
            x_max = min(int(mouth_coords[:, 0].max()) + 10, frame.shape[1])
            y_min = max(int(mouth_coords[:, 1].min()) - 10, 0)
            y_max = min(int(mouth_coords[:, 1].max()) + 10, frame.shape[0])

            mouth_crop    = gray[y_min:y_max, x_min:x_max]
            mouth_resized = cv2.resize(mouth_crop, (target_w, target_h))
            labels.append(f"Frame {frame_idx}")

        crops.append(mouth_resized)

    cap.release()

    # Plot grid of crops
    cols = 5
    rows = (len(crops) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = axes.flatten()

    for i, (crop, label) in enumerate(zip(crops, labels)):
        axes[i].imshow(crop, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(label, fontsize=8)
        axes[i].set_xlabel(f"{crop.shape[1]}×{crop.shape[0]}px", fontsize=7)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for j in range(len(crops), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Cropped Mouth Frames — Model Input ({target_w}×{target_h} grayscale)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    print(f"\n✅ {len(crops)} frames shown out of {total_frames} total.")
    print(f"   Each crop is {target_w}×{target_h} px grayscale — matching model input.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise face landmarks and/or mouth crops.")
    parser.add_argument("--video", required=False, default=None,
                        help="Path to your video file.")
    parser.add_argument("--mode", choices=["landmarks", "crops", "both"], default="both",
                        help="What to display (default: both).")
    parser.add_argument("--frames", type=int, default=6,
                        help="Number of frames to sample for landmarks view (default: 6).")
    args = parser.parse_args()

    VIDEO_PATH = args.video or r"C:\users\firas\onedrive\desktop\9rayaaaa\pfa\pfa2\files\RL tests\vids\bin blue at f four please.mpg"

    if args.mode in ("landmarks", "both"):
        visualize_landmarks(VIDEO_PATH, num_frames_to_show=args.frames)

    if args.mode in ("crops", "both"):
        visualize_mouth_crops(VIDEO_PATH, num_frames_to_show=15)
