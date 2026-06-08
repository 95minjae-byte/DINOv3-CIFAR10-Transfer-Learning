# =========================================================
# DINOv3 Transfer Learning for CIFAR-10
# train_dino.py
# =========================================================

# -----------------------------
# 기본 라이브러리 import
# -----------------------------
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
# CIFAR-10 이미지는 원래 32x32인데
# DINOv3 입력 크기에 맞게 224x224로 resize

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================================
# 2. CIFAR-10 Dataset 불러오기
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
# batch 단위로 이미지를 가져옴

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
# 4. DINOv3 + Linear Classifier 모델
# =========================================================

class DINOClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # -----------------------------
        # pretrained DINOv3 backbone
        # num_classes=0:
        # 마지막 classification head 제거
        # feature vector만 출력
        # -----------------------------
        self.backbone = timm.create_model(
            'vit_small_patch16_dinov3',
            pretrained=True,
            num_classes=0
        )

        # -----------------------------
        # backbone freeze
        # DINO는 학습하지 않음
        # -----------------------------
        for param in self.backbone.parameters():
            param.requires_grad = False

        # -----------------------------
        # feature dimension 확인
        # vit_small_patch16_dinov3는 보통 384 dim
        # -----------------------------
        feature_dim = self.backbone.num_features

        # -----------------------------
        # Linear classifier
        # CIFAR-10 → class 10개
        # -----------------------------
        self.classifier = nn.Linear(feature_dim, 10)

    # -----------------------------
    # forward
    # image → DINO → classifier
    # -----------------------------
    def forward(self, x):

        # backbone은 frozen 상태
        # gradient 계산 안 함
        with torch.no_grad():
            features = self.backbone(x)

        outputs = self.classifier(features)

        return outputs


# =========================================================
# 5. 모델 생성
# =========================================================

model = DINOClassifier().to(device)

# 모델 구조 출력
print(model)

# =========================================================
# 6. Loss Function
# =========================================================
# classification에서 가장 많이 사용

criterion = nn.CrossEntropyLoss()

# =========================================================
# 7. Optimizer
# =========================================================
# classifier만 학습

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# =========================================================
# 8. Training Loop
# =========================================================

epochs = 5

for epoch in range(epochs):

    # training mode
    model.train()

    running_loss = 0.0

    # batch 반복
    for images, labels in train_loader:

        # GPU 이동
        images = images.to(device)
        labels = labels.to(device)

        # gradient 초기화
        optimizer.zero_grad()

        # 예측
        outputs = model(images)

        # loss 계산
        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # optimizer update
        optimizer.step()

        running_loss += loss.item()

    # epoch loss 출력
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

# =========================================================
# 9. Evaluation
# =========================================================

model.eval()

correct = 0
total = 0
all_labels = []
all_predictions = []

# evaluation에서는 gradient 필요 없음
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # 가장 높은 score의 class 선택
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

# accuracy 계산
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

torch.save(model.state_dict(), "dino_model.pth")