"""
make_personal_dataset.py
------------------------
Records your own personalized LipNet-compatible dataset with ZERO manual
annotation effort. Just sit in front of your webcam and read sentences as
they appear on screen.

How it works:
    1. Shows a sentence on screen from a GRID-style sentence list.
    2. Starts recording your webcam automatically.
    3. You read the sentence aloud.
    4. Press SPACE to stop and save, or ESCAPE to discard and move on.
    5. The script saves the .mpg video AND generates the .align file
       automatically — because you told the script exactly what you said.

Output structure (drop-in compatible with your existing pipeline):
    personal_data/
    ├── s1/
    │   ├── session001.mpg
    │   ├── session002.mpg
    │   └── ...
    └── alignments/
        └── s1/
            ├── session001.align
            ├── session002.align
            └── ...

After recording, just update DATA_DIR in config.py to point to personal_data/s1
and ALIGN_DIR to personal_data/alignments/s1 — no other changes needed.

Requirements:
    pip install opencv-python

Run with:
    python make_personal_dataset.py
    python make_personal_dataset.py --sessions 50 --fps 25
    python make_personal_dataset.py --output my_data --list custom_sentences.txt
"""

import argparse
import os
import time
import random

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Sentence bank
# ─────────────────────────────────────────────────────────────────────────────
#
# These follow the GRID grammar pattern:
#   <verb> <colour> <preposition> <letter> <digit> <adverb>
#
# The model was trained on exactly this grammar. Recording sentences in this
# format gives you the best possible performance boost.
#
# The model's vocabulary is: a-z, space, and '?!123456789
# All words below use only those characters.

GRID_SENTENCES = [
    # verb   colour  prep   letter  digit  adverb
    "bin blue at a two now",
    "bin blue at b three please",
    "bin blue at c four soon",
    "bin blue at d five again",
    "bin blue at e six now",
    "bin blue at f seven please",
    "bin blue at g eight soon",
    "bin blue at h nine again",
    "bin green at a two now",
    "bin green at b three please",
    "bin green at c four soon",
    "bin green at d five again",
    "bin green at e six now",
    "bin green at f seven please",
    "bin green at g eight soon",
    "bin green at h nine again",
    "bin red at a two now",
    "bin red at b three please",
    "bin red at c four soon",
    "bin red at d five again",
    "bin white at a two now",
    "bin white at b three please",
    "bin white at c four soon",
    "bin white at d five again",
    "lay blue at a two now",
    "lay blue at b three please",
    "lay blue at c four soon",
    "lay blue at d five again",
    "lay green at a two now",
    "lay green at b three please",
    "lay green at c four soon",
    "lay green at d five again",
    "lay red at a two now",
    "lay red at b three please",
    "lay white at a two now",
    "lay white at b three please",
    "place blue at a two now",
    "place blue at b three please",
    "place blue at c four soon",
    "place green at a two now",
    "place green at b three please",
    "place green at c four soon",
    "place red at a two now",
    "place red at b three please",
    "place white at a two now",
    "place white at b three please",
    "set blue at a two now",
    "set blue at b three please",
    "set blue at c four soon",
    "set blue at d five again",
    "set green at a two now",
    "set green at b three please",
    "set green at c four soon",
    "set green at d five again",
    "set red at a two now",
    "set red at b three please",
    "set white at a two now",
    "set white at b three please",
]

# ─────────────────────────────────────────────────────────────────────────────
# Alignment generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_align_file(sentence: str, total_frames: int, fps: int) -> str:
    """
    Generate a synthetic .align file from a sentence and video metadata.

    The GRID .align format has one line per word:
        <start_frame> <end_frame> <word>

    Since we know exactly what was said (we showed the sentence on screen),
    we distribute frames evenly across words. The model only uses the character
    content of the words — the exact frame boundaries are only used during
    training for CTC alignment, and even approximate timings work fine.

    Args:
        sentence:      The spoken sentence string.
        total_frames:  Number of frames in the recorded video.
        fps:           Recording frame rate.

    Returns:
        String content of the .align file, ready to write to disk.
    """
    words           = sentence.strip().split()
    frames_per_word = total_frames // len(words)
    lines           = []

    # Leading silence
    lines.append(f"0 {frames_per_word // 2} sil")

    for i, word in enumerate(words):
        start = frames_per_word // 2 + i * frames_per_word
        end   = start + frames_per_word
        lines.append(f"{start} {end} {word}")

    # Trailing silence
    lines.append(f"{end} {total_frames} sil")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Recording engine
# ─────────────────────────────────────────────────────────────────────────────

def record_session(
    sentence: str,
    output_video_path: str,
    output_align_path: str,
    fps: int = 25,
    resolution: tuple = (640, 480),
    countdown_seconds: int = 3,
) -> bool:
    """
    Show the sentence on screen, record a video, save it and generate .align.

    Controls:
        SPACE  → stop recording and save
        ESCAPE → discard this recording and skip to next sentence

    Returns:
        True  if the recording was saved successfully.
        False if the user pressed ESCAPE to discard.
    """
    cap = cv2.VideoCapture(0)   # 0 = default webcam

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Make sure your camera is connected and not"
            " in use by another application."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc  = cv2.VideoWriter_fourcc(*"MPG1")   # .mpg compatible with GRID pipeline
    writer  = cv2.VideoWriter(output_video_path, fourcc, fps, (actual_w, actual_h))
    frames  = []
    saved   = False

    # ── Phase 1: Countdown ────────────────────────────────────────────────────
    start_countdown = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        elapsed = time.time() - start_countdown
        remaining = countdown_seconds - int(elapsed)

        if remaining > 0:
            _overlay_text(display, f"GET READY — {remaining}", (0, 200, 255), large=True)
            _overlay_text(display, f'Say: "{sentence}"', (255, 255, 255), y_offset=80)
            _overlay_text(display, "ESCAPE = skip", (150, 150, 150), y_offset=130)
        else:
            break   # countdown done

        cv2.imshow("LipNet Dataset Recorder", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESCAPE
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            return False

    # ── Phase 2: Recording ────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame.copy())
        writer.write(frame)

        display = frame.copy()
        _overlay_text(display, "● REC", (0, 0, 255), large=True)
        _overlay_text(display, f'Say: "{sentence}"', (255, 255, 255), y_offset=80)
        _overlay_text(display, "SPACE = done   |   ESCAPE = discard", (150, 150, 150), y_offset=130)
        cv2.imshow("LipNet Dataset Recorder", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:   # SPACE — save and move on
            saved = True
            break
        elif key == 27:  # ESCAPE — discard
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if saved and len(frames) > 0:
        # Generate and write the .align file
        align_content = generate_align_file(sentence, len(frames), fps)
        with open(output_align_path, "w") as f:
            f.write(align_content)
        print(f"  ✅ Saved: {os.path.basename(output_video_path)}  ({len(frames)} frames)")
        print(f"     Align: {os.path.basename(output_align_path)}")
    else:
        # Clean up the incomplete video file
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        print(f"  ⏭  Skipped.")

    return saved


def _overlay_text(
    frame: np.ndarray,
    text: str,
    color: tuple,
    y_offset: int = 0,
    large: bool = False,
) -> None:
    """Draw text centred at the top of the frame with a semi-transparent background."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 1.2 if large else 0.7
    thickness  = 3 if large else 2
    h, w       = frame.shape[:2]

    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - text_w) // 2
    y = 60 + y_offset

    # Semi-transparent dark box behind text
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def load_sentence_list(path: str) -> list:
    """Load sentences from a plain text file (one sentence per line)."""
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Record a personal LipNet dataset with auto-generated alignment files."
    )
    parser.add_argument("--output",   default="personal_data",
                        help="Root output folder (default: personal_data)")
    parser.add_argument("--sessions", type=int, default=len(GRID_SENTENCES),
                        help=f"Number of sentences to record (default: {len(GRID_SENTENCES)})")
    parser.add_argument("--fps",      type=int, default=25,
                        help="Recording frame rate (default: 25)")
    parser.add_argument("--list",     default=None,
                        help="Path to a custom sentence list (.txt, one per line)")
    parser.add_argument("--speaker",  default="s1",
                        help="Speaker ID subfolder name (default: s1)")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Countdown seconds before each recording (default: 3)")
    args = parser.parse_args()

    # ── Setup output folders ──────────────────────────────────────────────────
    video_dir = os.path.join(args.output, args.speaker)
    align_dir = os.path.join(args.output, "alignments", args.speaker)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(align_dir, exist_ok=True)

    # ── Load sentence list ────────────────────────────────────────────────────
    if args.list:
        sentences = load_sentence_list(args.list)
        print(f"Loaded {len(sentences)} custom sentences from: {args.list}")
    else:
        sentences = GRID_SENTENCES.copy()

    random.shuffle(sentences)
    sentences = sentences[:args.sessions]

    # ── Find next available session index ────────────────────────────────────
    existing = [f for f in os.listdir(video_dir) if f.endswith(".mpg")]
    session_idx = len(existing) + 1

    # ── Recording loop ────────────────────────────────────────────────────────
    total   = len(sentences)
    saved   = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"  LipNet Personal Dataset Recorder")
    print(f"  Target: {total} recordings → {video_dir}")
    print(f"  SPACE = save  |  ESCAPE = skip  |  Ctrl+C = quit")
    print(f"{'='*60}\n")

    for i, sentence in enumerate(sentences):
        session_name = f"session{session_idx:04d}"
        video_path   = os.path.join(video_dir, f"{session_name}.mpg")
        align_path   = os.path.join(align_dir,  f"{session_name}.align")

        print(f"[{i+1}/{total}]  Session: {session_name}")
        print(f"         Sentence: \"{sentence}\"")

        result = record_session(
            sentence          = sentence,
            output_video_path = video_path,
            output_align_path = align_path,
            fps               = args.fps,
            countdown_seconds = args.countdown,
        )

        if result:
            saved += 1
            session_idx += 1
        else:
            skipped += 1

        print()

    print(f"{'='*60}")
    print(f"  Done!  Saved: {saved}  |  Skipped: {skipped}")
    print(f"  Videos : {video_dir}")
    print(f"  Aligns : {align_dir}")
    print(f"{'='*60}")
    print()
    print("To use your new data, update config.py:")
    print(f'  DATA_DIR  = r"{os.path.abspath(video_dir)}"')
    print(f'  ALIGN_DIR = r"{os.path.abspath(align_dir)}"')


if __name__ == "__main__":
    main()
