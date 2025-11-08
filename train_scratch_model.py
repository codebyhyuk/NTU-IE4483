import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet import ResNet50

# ---------------------------
# 1) Variable Declaration
# ---------------------------
TRAIN_DIR = "/projects/448302/datasets/train"
VAL_DIR   = "/projects/448302/datasets/val"
LEARNING_RATE = 1e-4
EPOCH = 5
BATCH_SIZE = 32

# ---------------------------
# 2) Image Preprocess
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
### check
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# 3) Dataset / DataLoader
# ---------------------------
def load_data():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Classes:", train_dataset.classes)
    return train_loader, val_loader

# ---------------------------
# 4) Train Function
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ---------------------------
# 5) Validation Function
# ---------------------------
def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            _, predicted = torch.max(preds, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ---------------------------
# 6) Custom ResNet50 with Classification Header
# ---------------------------



# ---------------------------
# 7) Main Pipeline
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = load_data()

    model = ResNet50(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epochs = EPOCH

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save Model
    torch.save(model.state_dict(), "/checkpoints/resnet50_scratch.pth")
    print("Model Saved: resnet50_catsdogs.pth")


if __name__ == "__main__":
    main()
