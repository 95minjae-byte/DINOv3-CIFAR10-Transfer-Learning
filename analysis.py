# =========================================================
# analysis.py
# DINOv3 Analysis
# - Confusion Matrix
# - Failure Case Analysis
# - t-SNE Visualization
# =========================================================

# =========================================================
# 1. Import
# =========================================================

import torch
import torch.nn as nn
import timm

import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import seaborn as sns

# =========================================================
# 2. Device
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# =========================================================
# 3. Transform
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================================
# 4. CIFAR-10 Dataset
# =========================================================

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# CIFAR-10 class names
classes = test_dataset.classes

# =========================================================
# 5. DINO Classifier Model
# =========================================================

class DINOClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # pretrained DINOv3
        self.backbone = timm.create_model(
            'vit_small_patch16_dinov3',
            pretrained=True,
            num_classes=0
        )

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        feature_dim = self.backbone.num_features

        # linear classifier
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, x):

        # feature extraction
        with torch.no_grad():
            features = self.backbone(x)

        outputs = self.classifier(features)

        return outputs, features


# =========================================================
# 6. Load Trained Model
# =========================================================

model = DINOClassifier().to(device)

# IMPORTANT:
# train_dino.py에서 저장한 모델 불러오기
model.load_state_dict(torch.load("dino_model.pth"))

model.eval()

print("Model loaded successfully!")

# =========================================================
# 7. Prediction
# =========================================================

all_labels = []
all_predictions = []

# t-SNE용 feature 저장
all_features = []

# failure case 저장
failure_images = []
failure_true = []
failure_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs, features = model(images)

        _, predicted = torch.max(outputs, 1)

        # confusion matrix용 저장
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        # t-SNE용 feature 저장
        all_features.extend(features.cpu().numpy())

        # failure case 저장
        for i in range(len(labels)):

            if predicted[i] != labels[i]:

                failure_images.append(images[i].cpu())
                failure_true.append(labels[i].cpu().item())
                failure_pred.append(predicted[i].cpu().item())

# =========================================================
# 8. Confusion Matrix
# =========================================================

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

# =========================================================
# 9. Failure Case Analysis
# =========================================================

print("Showing Failure Cases...")

plt.figure(figsize=(12, 8))

# 실패 사례 6개 출력
num_examples = 6

for i in range(num_examples):

    plt.subplot(2, 3, i + 1)

    # tensor → numpy
    image = failure_images[i].permute(1, 2, 0).numpy()

    plt.imshow(image)

    true_label = classes[failure_true[i]]
    pred_label = classes[failure_pred[i]]

    plt.title(f"True: {true_label}\nPred: {pred_label}")

    plt.axis("off")

plt.tight_layout()

plt.show()

# =========================================================
# 10. t-SNE Visualization
# =========================================================

print("Running t-SNE...")

# numpy 변환
all_features = np.array(all_features)
all_labels = np.array(all_labels)

# 계산량 줄이기 위해 일부만 사용
sample_size = 2000

features_subset = all_features[:sample_size]
labels_subset = all_labels[:sample_size]

# t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

tsne_results = tsne.fit_transform(features_subset)

# plot
plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    tsne_results[:, 0],
    tsne_results[:, 1],
    c=labels_subset,
    cmap='tab10',
    s=10
)

plt.colorbar(scatter)

plt.title("t-SNE Visualization of DINOv3 Features")

plt.show()

print("Analysis completed!")