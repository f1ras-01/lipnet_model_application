"""
train_combined.py
-----------------
Fine-tunes the LipNet model on a combination of datasets:

    Mode A:  GRID + MIRACL-VC1    (--mode miracl)
    Mode B:  GRID + Personal data (--mode personal)
    Mode C:  MIRACL only          (--mode miracl --no-grid)
    Mode D:  Personal only        (--mode personal --no-grid)

Strategy:
    1. Load existing checkpoint (keeps what the model learned from GRID).
    2. Add new data on top.
    3. Train with a lower learning rate (0.00001 instead of 0.0001) to avoid
       catastrophic forgetting — the model adjusts rather than re-learns.

Run with:
    python train_combined.py --mode miracl   --miracl-root miraclvc1/
    python train_combined.py --mode personal --personal-root personal_data/
    python train_combined.py --mode miracl   --miracl-root miraclvc1/ --no-grid
"""

import argparse
import os

import tensorflow as tf

# ── GPU ───────────────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except (IndexError, RuntimeError):
    pass

# ── Imports ───────────────────────────────────────────────────────────────────
from config import CHECKPOINT_PATH, MODEL_DIR
from model import CTCLoss, build_model, get_callbacks

# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LipNet on combined datasets.")
    parser.add_argument("--mode",          choices=["miracl", "personal"], required=True)
    parser.add_argument("--miracl-root",   default="miraclvc1/",
                        help="Root folder of MIRACL dataset")
    parser.add_argument("--personal-root", default="personal_data/",
                        help="Root folder of personal dataset (from make_personal_dataset.py)")
    parser.add_argument("--no-grid",       action="store_true",
                        help="Don't include the GRID corpus (fine-tune on new data only)")
    parser.add_argument("--epochs",        type=int, default=50,
                        help="Number of additional epochs (default: 50)")
    parser.add_argument("--lr",            type=float, default=0.00001,
                        help="Fine-tuning learning rate (default: 0.00001 — lower than initial)")
    args = parser.parse_args()

    # ── Build train / test datasets ───────────────────────────────────────────
    train_datasets = []
    test_datasets  = []

    if not args.no_grid:
        from dataset import train as grid_train, test as grid_test
        train_datasets.append(grid_train)
        test_datasets.append(grid_test)
        print("✅ GRID corpus included.")

    if args.mode == "miracl":
        from data_loader_miracl import build_miracl_dataset
        miracl_train, miracl_test = build_miracl_dataset(args.miracl_root)
        train_datasets.append(miracl_train)
        test_datasets.append(miracl_test)
        print("✅ MIRACL-VC1 included.")

    elif args.mode == "personal":
        # Personal data uses the exact same folder structure as GRID,
        # so we just rebuild the pipeline pointing at the personal_data folder.
        import glob
        personal_video_dir = os.path.join(args.personal_root, "s1")
        personal_align_dir = os.path.join(args.personal_root, "alignments", "s1")

        if not os.path.isdir(personal_video_dir):
            raise FileNotFoundError(
                f"Personal video folder not found: {personal_video_dir}\n"
                "Run make_personal_dataset.py first."
            )

        # Temporarily override config paths for personal data
        import config
        original_data_dir  = config.DATA_DIR
        original_align_dir = config.ALIGN_DIR

        config.DATA_DIR  = personal_video_dir
        config.ALIGN_DIR = personal_align_dir

        # Reimport dataset with new paths
        import importlib
        import dataset as ds_module
        importlib.reload(ds_module)

        train_datasets.append(ds_module.train)
        test_datasets.append(ds_module.test)

        # Restore original config
        config.DATA_DIR  = original_data_dir
        config.ALIGN_DIR = original_align_dir

        n_vids = len(glob.glob(os.path.join(personal_video_dir, "*.mpg")))
        print(f"✅ Personal data included ({n_vids} videos).")

    if len(train_datasets) == 0:
        raise ValueError("No datasets selected. Check your arguments.")

    # ── Combine datasets ──────────────────────────────────────────────────────
    combined_train = train_datasets[0]
    combined_test  = test_datasets[0]
    for ds in train_datasets[1:]:
        combined_train = combined_train.concatenate(ds)
    for ds in test_datasets[1:]:
        combined_test  = combined_test.concatenate(ds)

    # Re-shuffle the combined training set so GRID and new data are mixed
    combined_train = combined_train.shuffle(1000, reshuffle_each_iteration=True)

    # ── Build model and load existing weights ─────────────────────────────────
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=CTCLoss,
    )

    if os.path.exists(CHECKPOINT_PATH + ".index"):
        model.load_weights(CHECKPOINT_PATH).expect_partial()
        print(f"✅ Loaded checkpoint: {CHECKPOINT_PATH}")
        print(f"   Fine-tuning with lr={args.lr} (lower than initial to avoid forgetting)")
    else:
        print("⚠️  No checkpoint found — training from scratch.")

    model.summary()

    # ── Train ─────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = get_callbacks(combined_test)

    print(f"\nFine-tuning for {args.epochs} epochs …\n")
    model.fit(
        combined_train,
        validation_data=combined_test,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"\nDone. Updated weights saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
