# =========================================================
# CNN Baseline for CIFAR-10
# cnn_baseline.py
# =========================================================

# -----------------------------
# 기본 라이브러리 import
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim

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
# CNN은 작은 이미지도 잘 처리 가능해서
# 원본 CIFAR-10 크기(32x32) 그대로 사용

transform = transforms.Compose([
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
# 4. Simple CNN Model
# =========================================================

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # -----------------------------
        # 첫 번째 convolution layer
        # -----------------------------
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # -----------------------------
        # 두 번째 convolution layer
        # -----------------------------
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # -----------------------------
        # pooling
        # 이미지 크기 줄임
        # -----------------------------
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # -----------------------------
        # activation function
        # -----------------------------
        self.relu = nn.ReLU()

        # -----------------------------
        # fully connected layer
        # 32x32
        # -> pool -> 16x16
        # -> pool -> 8x8
        # -----------------------------
        self.fc1 = nn.Linear(64 * 8 * 8, 512)

        # CIFAR-10 class 10개
        self.fc2 = nn.Linear(512, 10)

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, x):

        # Conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # flatten
        x = x.view(x.size(0), -1)

        # FC1
        x = self.fc1(x)
        x = self.relu(x)

        # FC2
        x = self.fc2(x)

        return x


# =========================================================
# 5. 모델 생성
# =========================================================

model = SimpleCNN().to(device)

print(model)

# =========================================================
# 6. Loss Function
# =========================================================

criterion = nn.CrossEntropyLoss()

# =========================================================
# 7. Optimizer
# =========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================================================
# 8. Training Loop
# =========================================================

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        # GPU 이동
        images = images.to(device)
        labels = labels.to(device)

        # gradient 초기화
        optimizer.zero_grad()

        # prediction
        outputs = model(images)

        # loss 계산
        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # parameter update
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

# =========================================================
# 9. Evaluation
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

        # 가장 높은 score 선택
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

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

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
