# DINOv3-CIFAR10-Transfer-Learning

## Overview

This project investigates the effectiveness of DINOv3 for transfer learning on the CIFAR-10 image classification task.

Three models are compared:

- DINOv3 + Linear Classifier (pretrained frozen backbone)
- Random ViT + Linear Classifier (without pretrained weights)
- CNN baseline trained from scratch

Additional analyses include:
- Confusion Matrix
- Failure Case Analysis
- t-SNE Visualization

---

## Repository Structure

```
DINOv3-CIFAR10-Transfer-Learning
│
├── train_dino.py                # Train DINOv3 + Linear classifier
├── random_ViT_classifier.py     # Train Random ViT baseline
├── cnn_baseline.py              # Train CNN baseline
├── analysis.py                  # Generate confusion matrix, failure cases, and t-SNE
├── requirements.txt             # Required Python packages
└── README.md
```

---

## Environment Setup

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset and Preprocessing

CIFAR-10 is automatically downloaded through torchvision.

Preprocessing is implemented inside each training script.

- DINOv3 and Random ViT:
  - Resize images from 32×32 to 224×224
  - Convert images to tensor

- CNN baseline:
  - Use original CIFAR-10 resolution (32×32)
  - Convert images to tensor

---

## Training

### 1. Train DINOv3 + Linear Classifier

```bash
python train_dino.py
```

This script trains only the linear classification head while keeping the pretrained DINOv3 backbone frozen.

The trained model will be saved as:

```
dino_model.pth
```

---

### 2. Train Random ViT Baseline

```bash
python random_ViT_classifier.py
```

---

### 3. Train CNN Baseline

```bash
python cnn_baseline.py
```

---

## Analysis

After training DINOv3, run:

```bash
python analysis.py
```

This script loads `dino_model.pth` and generates:
- Confusion Matrix
- Failure Case examples
- t-SNE feature visualization

---

## Experimental Configuration

- Dataset: CIFAR-10
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 5
- Loss Function: CrossEntropy Loss

---

## Expected Results

The following results were obtained under the above configuration:

| Model | Accuracy |
|-------|---------|
| DINOv3 + Linear | 90.19% |
| CNN Baseline | 70.79% |
| Random ViT + Linear | 28.51% |

