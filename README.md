# 🍃 Leaffliction

**Leaffliction** is a computer vision project that detects and classifies plant leaf diseases from images using a **Convolutional Neural Network (CNN)**. The pipeline covers the full machine learning workflow: dataset analysis, data augmentation, image transformation, training and prediction.

The dataset is composed of leaf images from two plants — **Apple** and **Grape** — split across 8 classes (healthy and diseased variants).

---

## 📚 Key concepts

### What is a CNN?

A **Convolutional Neural Network (CNN)** is a class of deep neural networks specifically designed to process data with a grid-like topology — typically images. Unlike fully-connected networks, a CNN exploits the **spatial structure** of an image by sliding small filters (kernels) across it to detect local patterns: edges, textures, shapes, and eventually high-level features like a leaf's veins or a disease spot.

A typical CNN is composed of three main building blocks:

- **Convolutional layers** — apply learnable filters across the image to produce *feature maps*. Each filter learns to recognize a specific pattern (e.g. a dark spot, a curved edge).
- **Pooling layers** — downsample feature maps (here `MaxPool2d`) to reduce spatial dimensions, lower computation cost and make the network more robust to small translations.
- **Fully-connected layers** — flatten the final feature maps and combine them to produce class scores.

<img width="655" height="363" alt="Capture d’écran 2026-05-24 à 14 49 53" src="https://github.com/user-attachments/assets/5f09bdef-22f2-4f9a-b3cd-80e2675c39ed" />

The model in this project stacks 3 convolutional blocks (`Conv2d` → `BatchNorm` → `ReLU` → `MaxPool`) followed by an adaptive pooling layer and two dense layers that output one of the 8 leaf classes. Training uses **Cross-Entropy Loss**, the **Adam** optimizer, and an **early-stopping** mechanism to prevent overfitting.


### Data augmentation

Deep learning models are data-hungry. **Augmentation** artificially expands the dataset by applying random transformations (flip, rotation, blur, contrast, crop, shear) to existing images. This balances classes that have fewer samples and helps the model generalize.

### Image transformation

To help the model focus on the relevant parts of a leaf, several **image transformations** are applied: Gaussian blur, Canny edge detection, Harris corner detection, Local Binary Patterns (LBP), leaf masking, and Region of Interest (ROI) extraction. These transformed images form the actual training set.

---

## Setup

1. Clone the repository

```bash
git clone https://github.com/MatLBS/leaffliction.git
cd leaffliction
```

2. Install dependencies with [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

---

## Commands

### 1. Analyze the dataset distribution

```bash
uv run srcs/distribution.py -src images
```

<img width="1132" height="482" alt="Capture d’écran 2026-05-23 à 17 20 15" src="https://github.com/user-attachments/assets/e4d26cb4-487f-4826-8891-10fcc5c3d84a" />


Displays a pie chart and a bar chart of the number of images per class for each plant.

### 2. Augment the dataset

```bash
uv run srcs/augmentation.py -src images
```

Balances the dataset by generating augmented variants (flipped, rotated, blurred, contrast, cropped, sheared) for classes with fewer samples. Output is written to `augmented_directory/`.

### 3. Transform the images

```bash
uv run srcs/transformation.py -src augmented_directory -dst transformed_directory
```

<img width="1174" height="590" alt="Capture d’écran 2026-05-23 à 17 21 47" src="https://github.com/user-attachments/assets/2f0939b7-69eb-45c1-b10e-7ff80079b3ed" />


Applies the full set of computer vision transformations to every image and saves them into `transformed_directory/`. This is the dataset used for training.

### 4. Train the model

```bash
uv run srcs/train.py -src transformed_directory --epochs 15
```

Trains the CNN, runs evaluation on the test split, saves the model as `models/model{epochs}.zip` and displays the training/validation loss curves.

You can also skip training and reuse a pre-trained model from the `models/` directory.

### 5. Predict the class of a leaf

Using an existing model:

```bash
uv run srcs/predict.py --model models/model20.zip --image images/Grape_Esca/image\ \(1\).JPG
```

Using your own trained model:

```bash
uv run srcs/predict.py --model models/XXX --image images/Grape_Esca/image\ \(1\).JPG
```

The original image and a random transformed version of it are displayed side by side, along with the predicted class.

<img width="947" height="489" alt="Capture d’écran 2026-05-23 à 17 24 43" src="https://github.com/user-attachments/assets/791a9919-a642-4844-be8d-33fc1f96db33" />

---

## CLI options

### `distribution.py`

| Option | Description |
|--------|-------------|
| `-src` | Path to the dataset folder to analyze (required) |

### `augmentation.py`

| Option | Description |
|--------|-------------|
| `-src` | Apply augmentation to every class in a directory and balance the dataset |
| `--specific` | Apply augmentation to a single image and display the result |

### `transformation.py`

| Option | Description |
|--------|-------------|
| `--specific` | Path to a single image — displays all transformations side by side |
| `-src` | Source directory containing the augmented dataset |
| `-dst` | Destination directory where transformed images will be saved |

### `train.py`

| Option | Description |
|--------|-------------|
| `-src` | Path to the transformed dataset directory (required) |
| `--epochs` | Number of training epochs — must be between 1 and 30 (default: 10) |

### `predict.py`

| Option | Description |
|--------|-------------|
| `--model` | Path to the model `.zip` file (required) |
| `--image` | Path to the image to classify (required) |

---

## Project architecture

```
leaffliction/
├── images/                       # Raw dataset (8 classes, Apple & Grape)
│   ├── Apple_Black_rot/
│   ├── Apple_healthy/
│   ├── Apple_rust/
│   ├── Apple_scab/
│   ├── Grape_Black_rot/
│   ├── Grape_Esca/
│   ├── Grape_healthy/
│   └── Grape_spot/
├── models/                       # Trained model checkpoints (.zip)
│   ├── model15.zip
│   └── model20.zip
├── srcs/
│   ├── distribution.py           # Dataset class distribution analysis
│   ├── augmentation.py           # Data augmentation pipeline
│   ├── transformation.py         # Computer vision transformations
│   ├── cnn.py                    # CNN model definition + training loop
│   ├── train.py                  # Training entry point
│   └── predict.py                # Prediction entry point
├── pyproject.toml                # Project dependencies (uv)
├── uv.lock
└── README.md
```
