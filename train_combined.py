"""
train_combined.py
-----------------
Fine-tune the LipNet model on a combination of datasets.

Modes:
  --mode miracl    ->  GRID + MIRACL-VC1  (or MIRACL only with --no-grid)
  --mode personal  ->  GRID + personal data recorded via make_personal_dataset.py

Strategy:
  1. Load existing checkpoint (preserves everything learned from GRID training).
  2. Combine with new dataset.
  3. Fine-tune at a lower learning rate (default 1e-5 vs training 1e-4) to
     avoid catastrophic forgetting of the original GRID knowledge.

All datasets now produce variable-resolution tensors — padded_batch handles
within-batch spatial variation identically to the main training pipeline.

Run with:
    python train_combined.py --mode miracl   --miracl-root miraclvc1/
    python train_combined.py --mode personal --personal-root personal_data/
    python train_combined.py --mode miracl   --miracl-root miraclvc1/ --augment
    python train_combined.py --mode miracl   --miracl-root miraclvc1/ --no-grid
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # suppress TF C++ INFO / WARNING / ERROR logs
os.environ["CUDA_VISIBLE_DEVICES"]  = ""        # tell TF not to look for CUDA at all on CPU-only machines

import argparse
import glob
import importlib

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from gpu_utils import setup_gpu
setup_gpu()

from config import CHECKPOINT_PATH, MODEL_DIR, TARGET_FRAMES
from model import CTCLoss, build_model, get_callbacks


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LipNet on GRID + MIRACL or GRID + personal data."
    )
    parser.add_argument(
        "--mode", choices=["miracl", "personal"], required=True,
        help="Which additional dataset to include."
    )
    parser.add_argument(
        "--miracl-root", default="miraclvc1/",
        help="Root folder of the MIRACL-VC1 dataset (default: miraclvc1/)."
    )
    parser.add_argument(
        "--personal-root", default="personal_data/",
        help="Root folder created by make_personal_dataset.py (default: personal_data/)."
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Exclude the GRID corpus — train on new data only."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of fine-tuning epochs (default: 50)."
    )
    parser.add_argument(
        "--lr", type=float, default=0.00001,
        help="Fine-tuning learning rate (default: 1e-5, lower than initial 1e-4)."
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable H-flip + frame jitter augmentation on the new dataset."
    )
    parser.add_argument(
        "--speaker", default="s1",
        help="Speaker subfolder name for personal data (default: s1)."
    )
    args = parser.parse_args()

    # ── padded_shapes shared by all datasets ──────────────────────────────────
    # None, None for H and W — adapts to whatever crop size each video produces
    _padded_shapes = ([TARGET_FRAMES, None, None, None], [40])

    train_datasets = []
    test_datasets  = []

    # ── GRID corpus ───────────────────────────────────────────────────────────
    if not args.no_grid:
        from dataset import train as grid_train, test as grid_test
        train_datasets.append(grid_train)
        test_datasets.append(grid_test)
        print("✅ GRID corpus included.")

    # ── MIRACL-VC1 ────────────────────────────────────────────────────────────
    if args.mode == "miracl":
        from data_loader_miracl import build_miracl_dataset
        miracl_train, miracl_test = build_miracl_dataset(
            archive_root = args.miracl_root,
            use_phrases  = True,
            augment      = args.augment,
        )
        train_datasets.append(miracl_train)
        test_datasets.append(miracl_test)
        print("✅ MIRACL-VC1 included.")

    # ── Personal data ─────────────────────────────────────────────────────────
    elif args.mode == "personal":
        personal_video_dir = os.path.join(args.personal_root, args.speaker)
        personal_align_dir = os.path.join(args.personal_root, "alignments", args.speaker)

        if not os.path.isdir(personal_video_dir):
            raise FileNotFoundError(
                f"Personal video folder not found: {personal_video_dir}\n"
                "Run make_personal_dataset.py first to record your data."
            )

        n_vids = len(glob.glob(os.path.join(personal_video_dir, "*.mpg")))
        if n_vids == 0:
            raise ValueError(f"No .mpg files found in {personal_video_dir}")

        # Temporarily override DATA_DIR and ALIGN_DIR in config so that
        # dataset.py builds a pipeline pointing at the personal data folder.
        import config as cfg
        original_data_dir  = cfg.DATA_DIR
        original_align_dir = cfg.ALIGN_DIR

        cfg.DATA_DIR  = personal_video_dir
        cfg.ALIGN_DIR = personal_align_dir

        import dataset as ds_mod
        importlib.reload(ds_mod)

        train_datasets.append(ds_mod.train)
        test_datasets.append(ds_mod.test)

        # Restore original paths so nothing else is affected
        cfg.DATA_DIR  = original_data_dir
        cfg.ALIGN_DIR = original_align_dir

        print(f"✅ Personal data included ({n_vids} videos from {personal_video_dir}).")

    if not train_datasets:
        raise ValueError("No datasets selected. Check your arguments.")

    # ── Combine datasets ──────────────────────────────────────────────────────
    combined_train = train_datasets[0]
    combined_test  = test_datasets[0]

    for ds in train_datasets[1:]:
        combined_train = combined_train.concatenate(ds)
    for ds in test_datasets[1:]:
        combined_test  = combined_test.concatenate(ds)

    # Shuffle combined training set so GRID and new data are interleaved
    combined_train = combined_train.shuffle(1000, reshuffle_each_iteration=True)

    # ── Build model and load checkpoint ───────────────────────────────────────
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.lr,
            clipnorm=1.0,     # clip gradient norm — prevents NaN loss from
                              # large padded batches mixing GRID + MIRACL sizes
        ),
        loss=CTCLoss,
    )

    if os.path.exists(CHECKPOINT_PATH + ".index"):
        model.load_weights(CHECKPOINT_PATH).expect_partial()
        print(f"✅ Checkpoint loaded: {CHECKPOINT_PATH}")
        print(f"   Fine-tuning with lr={args.lr}")
    else:
        # No checkpoint — training from scratch.
        # The --lr default (1e-5) is designed for fine-tuning only.
        # Override to 1e-4 (standard initial LR) to escape the inf-loss region.
        if args.lr < 1e-4:
            scratch_lr = 1e-4
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=scratch_lr,
                    clipnorm=1.0,
                ),
                loss=CTCLoss,
            )
            print(f"⚠️  No checkpoint — training from scratch at lr={scratch_lr}")
            print(f"   (overriding --lr {args.lr} which is too small for cold start)")
        else:
            print(f"⚠️  No checkpoint — training from scratch at lr={args.lr}")

    model.summary()

    # ── Train ─────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = get_callbacks(combined_test)

    print(f"\nFine-tuning for {args.epochs} epochs ...\n")
    model.fit(
        combined_train,
        validation_data=combined_test,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"\nDone. Updated weights saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
