# =========================================================
# Random ViT Baseline for CIFAR-10
# random_vit.py
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# GPU 사용 가능하면 GPU 사용
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# =========================================================
# 1. 데이터 전처리
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================================
# 2. CIFAR-10 Dataset
# =========================================================

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# =========================================================
# 3. DataLoader
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =========================================================
# 4. Random ViT + Linear Classifier
# =========================================================

class RandomViTClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=0
        )

        # backbone freeze
        for param in self.backbone.parameters():
            param.requires_grad = False

        feature_dim = self.backbone.num_features

        self.classifier = nn.Linear(
            feature_dim,
            10
        )

    def forward(self, x):

        with torch.no_grad():
            features = self.backbone(x)

        outputs = self.classifier(features)

        return outputs


model = RandomViTClassifier().to(device)

print(model)

# =========================================================
# 5. Loss Function
# =========================================================

criterion = nn.CrossEntropyLoss()

# =========================================================
# 6. Optimizer
# =========================================================

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# =========================================================
# 7. Training Loop
# =========================================================

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f}"
    )

# =========================================================
# 8. Evaluation
# =========================================================

model.eval()

correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

precision = precision_score(
    all_labels,
    all_predictions,
    average='macro'
)

recall = recall_score(
    all_labels,
    all_predictions,
    average='macro'
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average='macro'
)

print(f"Accuracy : {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

torch.save(model.state_dict(), "random_vit_model.pth")