# Fake Currency Detection (PyTorch, Simple CNN)

A lightweight image classification project that predicts whether an Indian
currency note image is **Real** or **Fake**, built with a small custom CNN
(no pretrained/transfer-learning models — just 3 conv blocks) in PyTorch.

## Project Structure

```
fake_currency_detection/
├── data/
│   ├── Real/
│   │   ├── 10/      <- put real 10-rupee note images here
│   │   ├── 20/
│   │   ├── 50/
│   │   ├── 100/
│   │   ├── 200/
│   │   ├── 500/
│   │   └── 2000/
│   └── Fake/
│       ├── 10/      <- put fake 10-rupee note images here
│       ├── 20/
│       ├── 50/
│       ├── 100/
│       ├── 200/
│       ├── 500/
│       └── 2000/
├── models/
│   └── cnn_model.py       <- SimpleCNN architecture
├── utils/
│   └── dataset.py          <- data loading + automatic 80/20 train/test split
├── saved_models/
│   └── best_model.pth      <- created after training
├── train.py                 <- training script
├── predict.py                <- inference script (single image)
├── check_dataset.py           <- verifies folder/image counts before training
└── requirements.txt
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add your dataset

**Just drop your images into `data/Real/<denomination>/` and
`data/Fake/<denomination>/`.** No manual train/test splitting needed —
the code does an 80/20 split automatically and internally every time you run
`train.py`.

```
data/Real/10/*.jpg
data/Real/20/*.jpg
data/Real/500/*.jpg
data/Fake/10/*.jpg
data/Fake/500/*.jpg
...
```

**Tip:** Try to keep Real and Fake roughly balanced in total count, and
include a good mix of denominations, lighting conditions, and camera angles —
this affects accuracy far more than the model architecture.

## 3. Verify your dataset (recommended before training)

```bash
python check_dataset.py
```

This prints how many images were found per class (Real/Fake) and per
denomination subfolder, so you can catch an empty folder or naming typo
before wasting a training run.

## 4. Train the model

```bash
python train.py --epochs 20 --batch_size 32 --lr 0.001
```

This will:
- Load all images from `data/Real` and `data/Fake`
- Automatically split them 80% train / 20% test (stratified, so both
  classes stay balanced in each split)
- Train the `SimpleCNN` model
- Save the best-performing model (by test accuracy) to `saved_models/best_model.pth`

Useful flags: `--epochs`, `--batch_size`, `--lr`, `--img_size`, `--test_split`, `--data_dir`, `--save_dir`.

## 5. Run predictions on a new image

```bash
python predict.py --image path/to/your/note.jpg
```

Example output:
```
Image: path/to/your/note.jpg
Prediction: FAKE
Confidence: 94.32%
All class probabilities:
  Fake: 94.32%
  Real: 5.68%
```

## Model Details

- **Architecture:** 3 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool),
  followed by 2 fully connected layers with dropout for regularization.
- **Input size:** 128x128 RGB images (configurable via `--img_size`).
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam with `ReduceLROnPlateau` scheduler (reduces LR when
  test accuracy plateaus).
- **Data split:** 80% train / 20% test, done automatically in `utils/dataset.py`
  (stratified by class - no manual file moving required).
- **Augmentation (training only):** random horizontal flip, small rotation,
  color jitter — helps generalize across different lighting/camera conditions.

## Notes on Accuracy

- This is a small CNN by design, so it trains fast even on CPU. If accuracy
  is too low once you have real data, the highest-leverage improvements,
  in order, are usually:
  1. **More and more varied data** (different lighting, angles, backgrounds, phone cameras)
  2. Increasing image resolution (`--img_size 224`)
  3. Adding a 4th conv block (more capacity)
  4. Switching to a pretrained backbone (e.g., ResNet18/MobileNetV2 fine-tuned)
     if you're open to a heavier model later
- Since currency has fine-grained security features (watermark, security thread,
  micro-lettering), make sure your training images are reasonably high-resolution
  and well-lit/focused — blurry phone photos will hurt accuracy the most.
