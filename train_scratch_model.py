import os
import torch
import wandb
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet import ResNet50

# ---------------------------
# 1) Variable Declaration
# ---------------------------
# TRAIN_DIR = "/projects/448302/datasets/train"
# VAL_DIR   = "/projects/448302/datasets/val"
# LEARNING_RATE = 1e-4
# EPOCH = 50
# BATCH_SIZE = 32

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        default="/projects/448302/datasets")
    parser.add_argument("--lr", type=float,
                        default=1e-4)
    parser.add_argument("--epochs", type=int,
                        default=50)
    parser.add_argument("--batch_size", type=int,
                        default=32)
    parser.add_argument("--save_dir", type=str,
                        default="./runs")
    parser.add_argument("--wandb", type=str,
                        default="_"
    )

    return parser.parse_args()


# ---------------------------
# 3) Dataset / DataLoader
# ---------------------------
def load_data(data_dir=None):
    if data_dir is None: raise ValueError
    train_dir = data_dir + "/train"
    val_dir = data_dir + "/val"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Classes:", train_dataset.classes)
    return train_loader, val_loader


def visualize(loss: list[float], val_acc: list[float], save_dir: str):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = list(range(1, 1+len(loss)))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Val Accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, val_acc, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Calculate EMA5 for val_acc
    val_acc_series = pd.Series(val_acc)
    ema5_val_acc = val_acc_series.ewm(span=5, adjust=False).mean()

    # Plot EMA5
    ema_color = 'skyblue' # A lighter shade of blue
    ax2.plot(epochs, ema5_val_acc, color=ema_color, linestyle='--', label='EMA5 Val Accuracy')


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training Loss and Validation Accuracy over Epochs')

    # Add legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='center right') # Changed legend location


    fname = save_dir + "/training_loss_val_acc.png"
    plt.savefig(fname)
    return

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

def set_seed(seed=42):
    random.seed(seed)                   
    np.random.seed(seed)                 
    torch.manual_seed(seed)              
    torch.cuda.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed)      

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random Seed Fixed: {seed}")

# ---------------------------
# 7) Main Pipeline
# ---------------------------
def main(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = load_data(args.data_dir)

    model = ResNet50(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs
    loss_hist = []
    val_acc_hist = []
    timestamp = (datetime.utcnow() + timedelta(hours=8)).strftime("%m%d%H%M%S")

    if args.wandb != "disabled":
        run_name = f"bs{args.batch_size}-lr{args.lr}-{timestamp}"
        run = wandb.init(
            project="IE4483",
            name=run_name,
            group="dogcat-exp",

            # Track hyperparameters and run metadata.
            config={
                "runtime": timestamp,
                "pretrained": False,
                "dataset": "dogcat",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
            },
        )


    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, val_loader, device)
        loss_hist.append(train_loss)
        val_acc_hist.append(val_acc)

        if args.wandb != "disabled": run.log({"Epoch": epoch, "Training_Loss": train_loss, "Validation_Accuracy": val_acc*100})
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save Model
    torch.save(model.state_dict(), "./checkpoints/resnet50_scratch.pth")
    print("Model Saved: resnet50_catsdogs.pth")

    # Visualization
    timestamp = (datetime.utcnow() + timedelta(hours=8)).strftime("%m%d%H%M%S")
    save_dir = f"./runs/{timestamp}"
    if os.path.exists(save_dir) is False: os.makedirs(save_dir)

    visualize(loss=loss_hist, val_acc=val_acc_hist, save_dir=save_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
