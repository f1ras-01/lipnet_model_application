# LipNet — Project File Structure

## Folder layout

```
lipnet/
├── config.py           ← All paths and constants (edit this first)
├── utils.py            ← Vocabulary & char↔num lookup layers
├── data_loader.py      ← Video/alignment loading functions
├── dataset.py          ← tf.data pipeline (train/test split)
├── model.py            ← Network architecture + loss + callbacks
├── train.py            ← Training script
├── predict.py          ← Predict on GRID dataset videos
├── predict_custom.py   ← Predict on YOUR OWN recorded videos  ← use this!
└── visualize.py        ← Debug: see landmarks and mouth crops
```

---

## What each file does and why it exists

### `config.py`
One place for every path and number used anywhere in the project.
Change `DATA_DIR`, `CHECKPOINT_PATH`, or `DLIB_MODEL_PATH` here and
every other file picks up the change automatically.

### `utils.py`
Defines the two Keras `StringLookup` layers:
- `char_to_num` — maps `'a'` → `1`, `'b'` → `2`, etc.
- `num_to_char` — inverse: maps `1` → `'a'`

Both are module-level singletons imported by all other files.

### `data_loader.py`
Three loaders:
- `load_video(path)` — GRID corpus only; uses a fixed pixel crop `[190:236, 80:220]`
- `load_alignments(path)` — reads `.align` files and returns encoded labels
- `load_data(path)` — combines both into `(frames, labels)`
- `load_custom_video(path)` — for YOUR videos; uses dlib to find the mouth dynamically

### `dataset.py`
Builds the `tf.data` pipeline:
`list_files → shuffle → map(load_data) → padded_batch(2) → prefetch`
Then splits into `train` (450 batches) and `test` (remainder).

### `model.py`
Defines the LipNet architecture as a Keras `Sequential` model:
3× Conv3D blocks → TimeDistributed Flatten → 2× BiLSTM → Dense(41, softmax)

Also contains `CTCLoss`, the `scheduler` function, and the `get_callbacks()` helper.

### `train.py`
Runs `model.fit()`. Import order:  
`config → dataset → model → compile → (optional load_weights) → fit`

### `predict.py`
Quick test on a GRID corpus video. Loads weights, runs one video through
the model, and prints ground truth vs. prediction side-by-side.

### `predict_custom.py` ← **This is your main script**
The only file you need to touch regularly. Set `VIDEO_PATH` to your
recorded video, run it, and get the predicted text. Uses `load_custom_video`
from `data_loader.py` which handles dlib face detection internally.

### `visualize.py`
Two diagnostic functions:
- `visualize_landmarks()` — full frame with green face box, blue landmark dots,
  red mouth dots and red dashed crop box
- `visualize_mouth_crops()` — the actual 46×140 grayscale tiles the model sees

Run this whenever a prediction looks wrong to check whether dlib is
detecting your face correctly.

---

## Workflow for testing a new video

```bash
# 1. Check that dlib is finding your face correctly
python visualize.py --video "path/to/your/video.mp4" --mode both

# 2. Run inference
python predict_custom.py --video "path/to/your/video.mp4"
```

That's it — no need to open the notebook or run any other cells.

---

## Workflow for training

```bash
# Just run train.py — everything else is imported automatically
python train.py
```

---

## First-time setup checklist

- [ ] Download `shape_predictor_68_face_landmarks.dat` and place it in this folder  
      (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [ ] Download the GRID corpus data and weights as described in the YouTube video
- [ ] Verify paths in `config.py` match your local folder structure
- [ ] Run `python utils.py` to confirm the vocabulary loads correctly
- [ ] Run `python dataset.py` to confirm the data pipeline works
- [ ] Run `python model.py` to print the model summary
