import os
import torch
import wandb
import random
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet import ResNet50


def parse_args():
    '''
    1) Parse the input arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="dogcat",
                        help="dataset")
    parser.add_argument("--data_dir", type=str,
                        default="/projects/448302/datasets",
                        help="path to the dataset directory; must have /train, /val, /test as child directory")
    parser.add_argument("--lr", type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("--epochs", type=int,
                        default=50,
                        help="training epoch")
    parser.add_argument("--batch_size", type=int,
                        default=32,
                        help="batch size")
    parser.add_argument("--save_dir", type=str,
                        default="./runs",
                        help="directory where the visualization will be saved")
    parser.add_argument("--wandb", type=str,
                        default="_",
                        help="whether to log using wandb, a visualization tool")
    return parser.parse_args()

def load_data(data_dir=None):
    '''
    2) Load Dataset / Construct DataLoader
    -   input
            data_dir: (str) path to the dataset directory
    -   output
            train_loader: (torch.DataLoader) DataLoader for the training dataset
            val_loader: (torch.DataLoader) DataLoader for the validation dataset
    '''
    if data_dir is None: raise ValueError
    train_dir = data_dir + "/train"
    val_dir = data_dir + "/val"

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

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def visualize(loss: list[float], val_acc: list[float], save_dir: str):
    '''
    3) Code for Visualization of the result using matplotlib
    -   input
            loss: (list[float]) list of the loss history across the training epoch
            vall_acc: (list[float]) list of the validation accuracy across the training epoch
            data_dir: (str) path to the saving directory (where to save the generated figure of visualization)
    '''
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = list(range(1, 1+len(loss)))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis to plot val accuracy and training loss concurrently

    color = 'tab:blue'
    ax2.set_ylabel('Val Accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, val_acc, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Calculate EMA5 for validation accuracy, to reduce a stochasticity in the performance
    val_acc_series = pd.Series(val_acc)
    ema5_val_acc = val_acc_series.ewm(span=5, adjust=False).mean()

    # Plot EMA5
    ema_color = 'skyblue'
    ax2.plot(epochs, ema5_val_acc, color=ema_color, linestyle='--', label='EMA5 Val Accuracy')
    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy over Epochs')

    # Add legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='center right')

    fname = save_dir + "/training_loss_val_acc.png"
    plt.savefig(fname)
    return

def train_one_epoch(model, loader, criterion, optimizer, device):
    '''
    4) Training the model one epoch using the provided inputs
    -   input
            model: (torch.NN.module) neural network model that is the target of training
            loader: (torch.DataLoader) train dataloader to provide the data and target for training
            criterion: (??) the objective loss function that the optimizer will minimize
            optimizer: (torch.optim) an optimizer that will be used to train the neural network model
            device: (str) a device that will be used for training (either CPU or CUDA-GPU)
    -   output
            total_loss: (float) the total loss across the epoch
    '''
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

def validate(model, loader, device):
    '''
    5) Valudate the model on the validation dataset
    -   input
            model: (torch.NN.module) neural network model that is the target of evaluation
            loader: (torch.DataLoader) eval dataloader to provide the data and target for evalution
            device: (str) a device that will be used for training (either CPU or CUDA-GPU)
    -   output
            correct: (float) the ratio that the prediction result is corret
    '''
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

def set_seed(seed=42):
    '''
    6) Fix the seed that is used for the run to ensure reproducibility
    -   input
            seed: (int) the seed number
    '''
    random.seed(seed)                   
    np.random.seed(seed)                 
    torch.manual_seed(seed)              
    torch.cuda.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed)      

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random Seed Fixed: {seed}")


def main(args):
    set_seed(42)

    # Define and create required training configurations
    train_loader, val_loader = load_data(args.data_dir)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2 if args.data="dogcat" else "cifar10"
    model       = ResNet50(num_classes=num_classes).to(device)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs      = args.epochs
    
    loss_hist = []
    val_acc_hist = []
    timestamp = (datetime.utcnow() + timedelta(hours=8)).strftime("%m%d%H%M%S")

    # For logging using WanDB tools
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

    # Start training over training epochs
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
    save_dir = f"{args.save_dir}/{timestamp}"
    if os.path.exists(save_dir) is False: os.makedirs(save_dir)
    visualize(loss=loss_hist, val_acc=val_acc_hist, save_dir=save_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)