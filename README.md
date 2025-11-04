# 🐶🐱 Super-Resolution Assisted Image Classification

## Dogs vs Cats — Transfer Learning + SRGAN

This project implements a two-stage deep learning pipeline using PyTorch to classify dog vs cat images and evaluates whether Super-Resolution GAN (SRGAN) improves classification performance.

## ✅ Project Overview

| Model | Training Input | Description |
|-------|----------------|-------------|
| Model A | Original images resized to 128×128 | Baseline transfer-learning binary classifier |
| SRGAN | HR = 128×128, LR = 32×32 | Trained to generate 128×128 high-resolution images from low-resolution inputs |
| Model B | SRGAN-generated 128×128 images | Same architecture as Model A, trained on SR images |

We compare Model A vs Model B on the same test split using:
- Accuracy
- F1-Score
- ROC-AUC

**Goal:** Determine whether Super-Resolution improves classification accuracy.

## 📂 Dataset

**Kaggle Dogs vs Cats**  
https://www.kaggle.com/c/dogs-vs-cats/data

**Files used:**
- `train/` → contains `cat.*` & `dog.*` labeled images
- `test/` → not used (Kaggle test set has no labels)

We perform our own 70% train / 30% test stratified split from the labeled Kaggle train set.

## 💻 Environment & Requirements

| Component | Version |
|-----------|---------|
| Python  | 3.8+ |
| PyTorch | 2.x |
| CUDA | Recommended |
| Other libraries | tqdm, numpy, matplotlib, scikit-learn, OpenCV |

**Install dependencies:**
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm scikit-learn opencv-python pillow albumentations
```

## 🔧 Project Structure

```
dogs-vs-cats/
├── train/                  # Kaggle labeled images
├── splits/
│   ├── train70.csv
│   └── test30.csv
├── models/
│   ├── modelA_best.pt
│   └── modelB_best.pt
├── data_sr/               # SRGAN-generated images
├── figures/               # Plots & examples
├── notebook.ipynb
└── README.md
```

## 🚀 Reproduction Steps

### 1️⃣ Data Preparation
- Place Kaggle `train/` images in project folder
- Run notebook cell to create `train70.csv` & `test30.csv`
- Stratified split ensures equal cats/dogs distribution

### 2️⃣ Preprocessing
- Resize all images to 128×128
- Train augmentations: flip, rotate, color jitter
- Normalize with ImageNet mean/std
- Show sample transformed images

### 3️⃣ Train Model A — Transfer Learning
- **Backbone:** ResNet18 (or VGG16 / MobileNetV2)
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** AdamW

**Procedure:**
1. Freeze backbone, train classifier head (warmup)
2. Unfreeze, fine-tune entire network
3. Save `modelA_best.pt`

### 4️⃣ Train SRGAN
- **HR images:** 128×128
- **LR images:** downsampled to 32×32

**Loss components:**
- Pixel loss (MSE)
- VGG perceptual loss
- GAN adversarial loss

**Training schedule:**

| Stage | Epochs |
|-------|--------|
| Pretrain Generator | 5 |
| Adversarial Training | >150 |

- Save generator weights every few epochs
- Show LR → Bicubic → SR → HR examples

### 5️⃣ Generate SR Training Dataset
- Run SRGAN on 70% training images
- Save outputs to `data_sr/`
- Create `train70_SR.csv`

### 6️⃣ Train Model B on SR Images
- Same architecture & hyperparameters as Model A
- Train on SR dataset only (or SR + original)
- Save `modelB_best.pt`

### 7️⃣ Evaluation
- Evaluate both models on the same untouched 30% test split

**Metrics recorded:**
- Accuracy
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC curves

## 📊 Results Summary

| Model | Accuracy | F1 | AUC |
|-------|----------|----|----|
| Model A (baseline) | XX% | XX | XX |
| Model B (SRGAN) | XX% | XX | XX |

*(Fill in after running experiments)*

Optional: Include qualitative examples showing SRGAN improvements or artifacts.

## ✅ Key Insights

- Transfer learning works well on 128×128 inputs
- SRGAN improves image detail (visual examples)
- Classification performance may improve/degrade depending on SRGAN quality

*(Complete after experiments)*

## 📎 References

- [SRGAN (Ledig et al., CVPR 2017)](https://arxiv.org/abs/1609.04802)
- [PyTorch Official Models](https://pytorch.org/vision/stable/models.html)
- [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

## 👨‍🏫 What This Project Demonstrates

- CNN Transfer Learning
- GAN-based Super-Resolution
- Evaluation across multiple metrics (Accuracy, F1, AUC)
- Proper train/test splits with no data leakage
- Reproducible ML workflow

## 🔁 Reproducibility Checklist

| Item | Status |
|------|--------|
| Dataset link | ✅ |
| Fix random seeds | ✅ |
| List dependencies | ✅ |
| Show code & configs | ✅ |
| Save train/test CSVs | ✅ |
| Save model checkpoints | ✅ |
| Plot transformations | ✅ |
| Plot SRGAN outputs | ✅ |
| Compare A vs B metrics | ✅ |